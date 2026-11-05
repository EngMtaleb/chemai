"""Phase 1B proof of concept - the numbers for the deck (poc_data.js) and the report's tables (report_tables.md), all read
from the run's own files (out/). Nothing is typed by hand.

    python3 -B make_poc_data.py <deck folder>
"""
import json
import sys

import numpy as np
import pandas as pd

import poc_lib as P

L = P.L
OUT = P.HERE / "out"
KEY = ["plant", "record", "test", "target", "h", "block", "held_out", "train", "t"]


def read(name):
    return pd.read_csv(OUT / name, keep_default_na=False, na_values=[""])


PROVISIONAL = "--provisional" in sys.argv     # layout tests only: whatever pieces exist; the deck refuses it (POC.final)


def parts(kind):
    fs = sorted((OUT / "trees_parts").glob(f"P?_{kind}.csv"))
    return pd.concat([pd.read_csv(f, keep_default_na=False, na_values=[""]) for f in fs], ignore_index=True)


def predictions():
    if PROVISIONAL:
        tr = pd.concat([parts("record1"), parts("record2")], ignore_index=True)
        full = tr.groupby("plant").record.nunique()
        tr = tr[tr.plant.isin(full[full == 2].index)]
    else:
        tr = read("pred_trees.csv")
    tr["held_out"] = tr["held_out"].fillna("")
    methods = list(P.METHODS)
    if (OUT / "pred_nets.csv").exists():
        n = read("pred_nets.csv")
        n["held_out"] = n["held_out"].fillna("")
        m = tr.merge(n[KEY + ["now", "y", "LSTM", "RNN"]], on=KEY, how="left", suffixes=("", "_n"), validate="one_to_one")
        assert m["LSTM"].notna().all() and m["RNN"].notna().all(), "a tree row has no network forecast"
        assert np.allclose(m.now, m.now_n) and np.allclose(m.y, m.y_n), "the networks saw other rows"
        tr = m.drop(columns=["now_n", "y_n"])
        methods += ["LSTM", "RNN"]
    return tr, methods


def importance():
    out = {}
    for kind, col in (("groups", "group"), ("columns", "column")):
        d = parts(f"importance_{kind}") if PROVISIONAL else read(f"importance_{kind}.csv")
        d["inc"] = (d.mae_permuted_pp - d.mae_pp) * d.n
        d["base"] = d.mae_pp * d.n
        g = d.groupby(col).agg(inc=("inc", "sum"), n=("n", "sum"), base=("base", "sum"))
        reps = d.groupby(col).repeat.nunique().iloc[0]
        g["increase_pp"] = g.inc / g.n                 # repeats are rows too: the n-weighted mean over plants, blocks, repeats
        g["base_pp"] = g.base / g.n
        g["increase_pct"] = 100 * g.increase_pp / g.base_pp
        out[kind] = g[["increase_pp", "base_pp", "increase_pct"]].sort_values("increase_pp", ascending=False).round(4)
        assert reps == 5
    return out


def main():
    deck = next((a for a in sys.argv[1:] if not a.startswith("--")), None)
    t, c = P.load()
    pred, methods = predictions()
    sc = P.score_table(pred, methods, by=("record", "test", "target", "h"))
    pp = P.score_table(pred, methods, by=("plant", "record", "test", "target", "h"))
    # the negative control: XGBoost (and the hybrid) on record 2 as run, without SDI and turbidity, with them shifted
    ctrl = parts("control") if PROVISIONAL else read("pred_control.csv")
    ctrl["held_out"] = ctrl["held_out"].fillna("")
    main_r2 = pred[(pred.record == "record 2") & (pred.test == "rolling") & pred.h.isin([7, 14])]
    both = pd.concat([main_r2[KEY + ["now", "y"] + P.METHODS], ctrl[KEY + ["now", "y"] + P.METHODS]], ignore_index=True)
    csc = P.score_table(both, ["XGBoost", "hybrid"], by=("record", "target", "h"))
    piv = csc[csc.method == "XGBoost"].pivot_table(index=["target", "h"], columns="record", values="skill_pct")
    piv["shifted - without"] = piv["record 2, SDI and turbidity shifted"] - piv["record 2 without SDI and turbidity"]
    piv["as run - without"] = piv["record 2"] - piv["record 2 without SDI and turbidity"]
    control_ok = bool((piv["shifted - without"].abs() <= 2.0).all())
    imp = importance()
    shifts = ctrl.groupby("plant").shift_days.first().to_dict()
    # the control's investigation (when it missed): the missed cell re-run with three sets of five seeds (ctrl_check2.py),
    # and with the yearly cycle taken out before the same shift (ctrl_check3.py) - read from their own files
    inv = None
    if not control_ok:
        f2, f3 = OUT / "ctrl_check2_flux_pct_14.csv", OUT / "ctrl_check3_flux_pct_14.csv"
        assert f2.exists() and f3.exists(), "the control missed: its investigation must run first"
        d = pd.concat([pd.read_csv(f2), pd.read_csv(f3)], ignore_index=True)
        sk = d.groupby(["seeds", "variant"]).apply(lambda g: 100 * (1 - np.abs(g.y - g.XGBoost).mean() / np.abs(g.y - g.now).mean()),
                                                   include_groups=False).unstack()
        decl = piv.loc[("flux_pct", 14)]
        assert abs(sk.loc["0-4", "shifted"] - decl["record 2, SDI and turbidity shifted"]) < 1e-6 and \
            abs(sk.loc["0-4", "without"] - decl["record 2 without SDI and turbidity"]) < 1e-6, "step 2 does not reproduce the run"
        missed = [f"{tg}|{h}" for (tg, h), row in piv.iterrows() if abs(row["shifted - without"]) > 2.0]
        inv = {"cell": "flux_pct|14", "missed": missed,
               "seeds": {k: {"without": round(float(r["without"]), 3), "shifted": round(float(r["shifted"]), 3),
                             "no_cycle": round(float(r["shifted, no yearly cycle"]), 3),
                             "gap": round(float(r["shifted"] - r["without"]), 3),
                             "gap_no_cycle": round(float(r["shifted, no yearly cycle"] - r["without"]), 3)}
                         for k, r in sk.reindex(["0-4", "5-9", "10-14"]).iterrows()}}
        assert missed == ["flux_pct|14"], f"the investigation covers flux 14 days only; missed: {missed}"

    # what the deck shows: the rolling test and the unseen train, each target and horizon, every method, both records
    def cell(r):
        # four decimals: the deck rounds once, to what it shows (a stored 4.885 would show as 4.88 where the report says 4.89)
        return {"mae": round(r.MAE_pp, 4), "nc": round(r.MAE_no_change_pp, 4), "skill": round(r.skill_pct, 3),
                "blocks": [round(r[f"skill_b{b}"], 3) for b in range(4)], "pass": bool(r.passes), "n": int(r.n)}
    S = {}
    for r in sc.itertuples(index=False):
        S.setdefault(r.test, {}).setdefault(r.target, {}).setdefault(str(r.h), {}).setdefault(r.method, {})[r.record] = cell(
            pd.Series(r._asdict()))
    passes = sc[sc.passes].groupby(["record", "method"]).size().to_dict()
    cells = sc.groupby(["record", "method"]).size().to_dict()
    # the table page: a few days of one train, as logged
    sample = t[(t.plant == "P1") & (t.train == "A")].sort_values("day")
    sample = sample[sample.day >= 600].head(6)
    info = json.loads((P.DATA / "plants.json").read_text())
    pairs = {}
    for p in sorted(pred.plant.unique()):
        d, led, _ = P.plant(t, c, p)
        days = P.build_days(d, led)
        pairs[p] = days[days.valid].groupby("h").size().to_dict()
    POC = {
        "plants": [{"id": x["plant"], "law": x["law"], "seed": x["seed"], "rows": x["rows"], "cleanings": x["cleanings"],
                    "recorded": x["recorded_share"]} for x in info["plants"]],
        "rows": int(len(t)), "columns": [col for col in t.columns if col not in ("plant", "law", "train", "day")],
        "groups": {k: v for k, v in info["groups"].items()},
        "days": 772, "trains": 3,
        "pairs": {p: {str(h): int(n) for h, n in v.items()} for p, v in pairs.items()},
        "sample": json.loads(sample.drop(columns=["plant", "law"]).to_json(orient="records")),
        "methods": methods, "scores": S,
        "passes": {f"{r}|{m}": int(n) for (r, m), n in passes.items()},
        "cells": {f"{r}|{m}": int(n) for (r, m), n in cells.items()},
        "control": {"ok": control_ok, "shift_days": {k: int(v) for k, v in shifts.items()}, "investigation": inv,
                    "skill": {f"{tg}|{h}": {k: round(float(v), 3) for k, v in row.items()} for (tg, h), row in piv.iterrows()}},
        "importance": {"groups": imp["groups"].reset_index().to_dict(orient="records"),
                       "columns": imp["columns"].head(8).reset_index().to_dict(orient="records")},
        "nets": "LSTM" in methods,
        "final": bool(not PROVISIONAL and "LSTM" in methods and sorted(pred.plant.unique()) == sorted(P.PLANTS)),
        "plants_in": sorted(pred.plant.unique()),
    }
    txt = "/* made by make_poc_data.py from the proof of concept's own files; nothing typed by hand */\nconst POC = " + \
        json.dumps(POC, ensure_ascii=False, separators=(",", ":")) + ";\n"
    (OUT / "poc_data.js").write_text(txt)
    if deck:
        (P.Path(deck) / "poc_data.js").write_text(txt)
    # the report's tables
    lines = ["## Scores, pooled over the three plants", ""]
    for test in ("rolling", "unseen_train"):
        for tg in L.TARGETS:
            sub = sc[(sc.test == test) & (sc.target == tg)]
            lines += [f"### {test} · {tg} · MAE in points (skill against no change, %; ✓ = passes the bar)", "",
                      "| method | record | 1 d | 3 d | 7 d | 14 d |", "|---|---|---|---|---|---|"]
            for m in methods:
                for rec in ("record 1", "record 2"):
                    if m in ("no change", "fouling law") and rec == "record 2":
                        continue
                    row = []
                    for h in L.HORIZONS:
                        r = sub[(sub.method == m) & (sub.record == rec) & (sub.h == h)].iloc[0]
                        row.append(f"{r.MAE_pp:.2f} ({r.skill_pct:+.1f}{' ✓' if r.passes else ''})")
                    lines.append(f"| {m} | {rec if m not in ('no change', 'fouling law') else 'both'} | " + " | ".join(row) + " |")
            lines.append("")
    lines += ["## Each plant (rolling test, ΔP, 7 days): skill against no change, %", "",
              "| plant | method | record 1 | record 2 |", "|---|---|---|---|"]
    for p in sorted(pred.plant.unique()):
        for m in methods[2:]:
            v = {rec: pp[(pp.plant == p) & (pp.record == rec) & (pp.test == "rolling") & (pp.target == "dp_pct") & (pp.h == 7)
                         & (pp.method == m)].iloc[0] for rec in ("record 1", "record 2")}
            lines.append(f"| {p} | {m} | {v['record 1'].skill_pct:+.1f}{' ✓' if v['record 1'].passes else ''} | "
                         f"{v['record 2'].skill_pct:+.1f}{' ✓' if v['record 2'].passes else ''} |")
    lines += ["", "## The negative control (XGBoost, rolling test): skill against no change, %", "",
              piv.round(2).to_markdown(), "", f"Control passes (|shifted − without| ≤ 2 points everywhere): {control_ok}",
              f"Shifts (days): {shifts}", "", "## Which columns XGBoost used (record 2, rolling, ΔP 7 days): MAE increase when "
              "permuted, points", "", imp["groups"].to_markdown(), "", "Top columns (descriptive):", "",
              imp["columns"].head(10).to_markdown(), ""]
    (OUT / "report_tables.md").write_text("\n".join(lines))
    sc.to_csv(OUT / "scores_pooled.csv", index=False)
    pp.to_csv(OUT / "scores_per_plant.csv", index=False)
    csc.to_csv(OUT / "scores_control.csv", index=False)
    print("\n".join(lines[:60]))


if __name__ == "__main__":
    main()
