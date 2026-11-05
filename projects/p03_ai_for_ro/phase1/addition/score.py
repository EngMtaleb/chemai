"""Phase 3 addition - score the new methods exactly as declared, next to the six methods' saved numbers.

    python3 -B score.py

Joins the saved phase-3 forecasts (phase3/out, read only) with the new ones on the same test pairs, checks that the pairs,
today's values and truths agree, scores every method with p3lib.scores / p3lib.verdict (unchanged), checks that the six
methods' re-computed scores equal the saved score files, and writes scores_*_addition.csv and report_tables.md.
"""
import json

import numpy as np
import pandas as pd

import p3add as A
L = A.L
KEY = ["target", "h", "block", "train", "t"]
ALL = L.METHODS + A.NEW
NAMES = {"rolling": "rolling origin", "unseen_train": "unseen train"}


def joined(test_name):
    sv = pd.read_csv(A.OUT3 / f"pred_{test_name}.csv")
    pr = pd.read_csv(A.HERE / "pred_prophet.csv")
    nn = pd.read_csv(A.HERE / f"pred_nets_{test_name}.csv", keep_default_na=False, na_values=[""])
    m = sv.merge(pr, on=["train", "t", "target", "h"], how="left", validate="1:1")
    m = m.merge(nn, on=KEY, how="left", suffixes=("", "_nets"), validate="1:1")
    assert len(m) == len(sv) and not m[A.NEW].isna().any().any(), "a test pair lacks a new forecast"
    assert np.array_equal(m.now.to_numpy(), m.now_nets.to_numpy()) and np.array_equal(m.y.to_numpy(), m.y_nets.to_numpy())
    return m


def forecast_scores(p):
    rows = []
    for (c, h), g in p.groupby(["target", "h"]):
        pooled = L.scores(g.y.to_numpy(), g.now.to_numpy(), {m: g[m].to_numpy() for m in ALL})
        blocks = {b: L.scores(gb.y.to_numpy(), gb.now.to_numpy(), {m: gb[m].to_numpy() for m in ALL})
                  for b, gb in g.groupby("block")}
        for m in ALL:
            bs = [blocks[b][m]["skill_pct"] for b in sorted(blocks)]
            rows.append({"target": c, "h": h, "method": m, "source": "saved phase 3" if m in L.METHODS else "addition",
                         "n": len(g), **pooled[m], **{f"skill block {b + 1}": s for b, s in enumerate(bs)},
                         "blocks positive": int(sum(s > 0 for s in bs)),
                         "beats no change": L.verdict(pooled[m]["skill_pct"], bs) if m != "no change" else None})
    return pd.DataFrame(rows)


def check_saved(fs, test_name):
    sv = pd.read_csv(A.OUT3 / f"scores_{test_name}.csv")
    a = fs[fs.method.isin(L.METHODS)].set_index(["target", "h", "method"]).sort_index()
    b = sv.set_index(["target", "h", "method"]).sort_index()
    cols = ["MAE_pp", "skill_pct", "R2_change"] + [f"skill block {i + 1}" for i in range(4)]
    diff = float(np.nanmax(np.abs(a[cols].to_numpy(float) - b.loc[a.index, cols].to_numpy(float))))
    as_text = lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(bool(v))   # no verdict for no change
    v_a = a["beats no change"].map(as_text)
    v_b = b.loc[a.index, "beats no change"].map(as_text)
    return diff, bool((v_a == v_b).all())


def fmt_skill(x):
    return f"{x:+.1f}"


def tables(fs, test_name):
    out = []
    for c in L.TARGETS:
        f = fs[fs.target == c]
        base = f[f.method == "no change"].set_index("h").MAE_pp
        lab = "ΔP (`dp_pct`)" if c == "dp_pct" else "flux loss (`flux_pct`)"
        out.append(f"#### {lab} — {NAMES[test_name]}\n")
        out.append("*No change* MAE, percentage points: " + ", ".join(f"{h} d {v:.2f}" for h, v in base.items()) + "\n")
        out.append("| method | " + " | ".join(f"{h} d: skill % (blocks > 0) | bar" for h in L.HORIZONS) + " |")
        out.append("|---|" + "---|---|" * len(L.HORIZONS))
        for m in ALL[1:]:
            cells = []
            for h in L.HORIZONS:
                r = f[(f.method == m) & (f.h == h)].iloc[0]
                cells.append(f"{fmt_skill(r.skill_pct)} ({r['blocks positive']}/4) | {'**PASS**' if r['beats no change'] else 'fail'}")
            name = m if m in L.METHODS else f"**{m}** (new)"
            out.append(f"| {name} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


def detail(fs, test_name):
    out = []
    for c in L.TARGETS:
        for h in L.HORIZONS:
            f = fs[(fs.target == c) & (fs.h == h)]
            lab = "ΔP" if c == "dp_pct" else "flux loss"
            out.append(f"**{lab}, {h} d ahead, {NAMES[test_name]}** (n = {f.n.iloc[0]})\n")
            out.append("| method | MAE, pp | skill % | block 1 | block 2 | block 3 | block 4 | R² of the change | bar |")
            out.append("|---|---|---|---|---|---|---|---|---|")
            for _, r in f.iterrows():
                bar = "—" if r.method == "no change" else ("**PASS**" if r["beats no change"] else "fail")
                name = r.method if r.method in L.METHODS else f"**{r.method}** (new)"
                out.append(f"| {name} | {r.MAE_pp:.2f} | {fmt_skill(r.skill_pct)} | "
                           + " | ".join(fmt_skill(r[f'skill block {i + 1}']) for i in range(4))
                           + f" | {r.R2_change:.2f} | {bar} |")
            out.append("")
    return "\n".join(out)


def main():
    res, md_over, md_det = {}, [], []
    for test_name in ("rolling", "unseen_train"):
        p = joined(test_name)
        fs = forecast_scores(p)
        diff, same_verdicts = check_saved(fs, test_name)
        print(f"[{test_name}] six saved methods re-scored from the saved forecasts: largest difference from the saved scores "
              f"{diff:.1e}, same verdicts: {same_verdicts}")
        assert diff < 1e-9 and same_verdicts
        fs.to_csv(A.HERE / f"scores_{test_name}_addition.csv", index=False)
        res[test_name] = fs.to_dict("records")
        md_over.append(tables(fs, test_name))
        md_det.append(detail(fs, test_name))
        new = fs[fs.method.isin(A.NEW)]
        print(new.pivot_table(index=["target", "method"], columns="h", values="skill_pct").round(1).to_string())
        print("passes among the new methods:", new[new["beats no change"] == True][["target", "h", "method"]].values.tolist() or "none")
    (A.HERE / "report_tables.md").write_text("### Overview\n\n" + "\n".join(md_over) + "\n### Every horizon in detail\n\n"
                                             + "\n".join(md_det))
    (A.HERE / "summary_addition.json").write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


if __name__ == "__main__":
    main()
