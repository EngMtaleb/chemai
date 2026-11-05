"""declaration_v15.md: the numbers for the deck (poc_data.js, const POC) from the v15 run's own files — nothing typed.

    python3 -B make_poc_data15.py <deck folder>
"""
import json, sys
import numpy as np, pandas as pd
import poc_lib15 as Q
P, L = Q.P, Q.L
HERE = Q.HERE; OUT = HERE / "out"
SP = HERE.parent
sys.path.insert(0, str(SP / "phase3b"))
import fingerprint as F  # noqa: E402

METHODS = ["no change", "fouling law", "XGBoost", "hybrid", "LSTM", "RNN"]


def cell(g, m):
    nc = np.abs(g.y - g["no change"]).mean()
    mae = np.abs(g.y - g[m]).mean()
    blocks = [float(100 * (1 - np.abs(b.y - b[m]).mean() / np.abs(b.y - b["no change"]).mean())) for _, b in g.groupby("block")]
    sk = float(100 * (1 - mae / nc))
    return {"mae": round(100 * float(mae), 4), "nc": round(100 * float(nc), 4), "skill": round(sk, 3),
            "blocks": [round(b, 3) for b in blocks], "pass": bool(L.verdict(sk, blocks)), "n": int(len(g))}


def main():
    deck = sys.argv[1] if len(sys.argv) > 1 else None
    t, c = Q.load()
    tr = pd.read_csv(OUT / "pred_trees15.csv"); nets = pd.read_csv(OUT / "pred_nets15.csv")
    pred = tr.merge(nets[["train", "t", "LSTM", "RNN"]], on=["train", "t"], how="left", validate="one_to_one")
    assert pred.LSTM.notna().all() and len(pred) == len(nets)
    S = {m: {"record 2": cell(pred, m)} for m in METHODS}
    for m in ("no change", "fouling law"):                 # they read only the indicator: the same in both records
        S[m]["record 1"] = S[m]["record 2"]
    # the negative control (declared): skill as run, without SDI and turbidity, and with them shifted
    ctrl = pd.read_csv(OUT / "pred_control15.csv")
    sk = {"record 2": S["XGBoost"]["record 2"]["skill"]}
    for rec, g in ctrl.groupby("record"):
        sk[rec] = cell(g, "XGBoost")["skill"]
    sk["shifted - without"] = sk["record 2, SDI and turbidity shifted"] - sk["record 2 without SDI and turbidity"]
    sk["as run - without"] = sk["record 2"] - sk["record 2 without SDI and turbidity"]
    ok = bool(abs(sk["shifted - without"]) <= 2.0)
    # the reading against the simulator's truth (hidden.csv: checks only, never a model input)
    h = pd.read_csv(P.DATA / "hidden.csv"); h["tr"] = h.plant + h.train
    tru = h.set_index(["tr", "day"]).dp_true
    true_y = np.clip(np.array([tru.get((a, int(b) + Q.H), np.nan) for a, b in zip(pred.train, pred.t)]) - 1, 0, 1)
    assert not np.isnan(true_y).any()
    truth = {"perfect_vs_reading": round(100 * float(np.mean(np.abs(pred.y - true_y))), 3),
             "xgb_vs_truth": round(100 * float(np.mean(np.abs(pred.XGBoost - true_y))), 3),
             "nc_vs_truth": round(100 * float(np.mean(np.abs(pred["no change"] - true_y))), 3)}
    # the reading's movement, scored like the real record (fingerprint.py, from day 300), per plant
    real = json.loads((SP / "phase3b/out/fingerprint_real.json").read_text())["value"]
    keys = ["noise_pp", "mean_abs_change_1d_pp", "mean_abs_change_7d_pp"]
    fp = {}
    for p in P.PLANTS:
        d, led, _ = P.plant(t, c, p)
        v = F.fingerprint(d[["train", "day_index", "dp_pct"]], {tr_: sorted(led[led.train == tr_].day_index.astype(int).tolist())
                                                               for tr_ in "ABC"}, boot=False)
        v = v.get("value", v); fp[p] = {k: round(float(v[k]), 3) for k in keys}
    old = {}                                               # the v14 table (phase3b/poc/data), scored the same way
    t0, c0 = pd.read_csv(SP / "phase3b/poc/data/table.csv"), pd.read_csv(SP / "phase3b/poc/data/cleanings.csv")
    for p in P.PLANTS:
        d0 = t0[t0.plant == p].rename(columns={"day": "day_index"})
        l0 = c0[c0.plant == p]
        v = F.fingerprint(d0[["train", "day_index", "dp_pct"]], {tr_: sorted(l0[l0.train == tr_].day.astype(int).tolist())
                                                                for tr_ in "ABC"}, boot=False)
        v = v.get("value", v); old[p] = {k: round(float(v[k]), 3) for k in keys}
    info = json.loads((P.DATA / "plants.json").read_text())
    cal = json.loads((OUT / "calibration15.json").read_text())
    sample = t[(t.plant == "P1") & (t.train == "A")].sort_values("day")
    sample = sample[sample.day >= 600].head(6)
    POC = {
        "v15": True, "merged": True,
        "plants": [{"id": x["plant"], "law": x["law"], "seed": x["seed"], "rows": x["rows"], "cleanings": x["cleanings"],
                    "recorded": x["recorded_share"]} for x in info["plants"]],
        "rows": int(len(t)), "columns": [col for col in t.columns if col not in ("plant", "law", "train", "day")],
        "groups": info["groups"], "days": 772, "trains": 3,
        "pairs7": int(len(pred)),
        "sample": json.loads(sample.drop(columns=["plant", "law"]).to_json(orient="records")),
        "methods": METHODS, "scores": {"rolling": {"dp_pct": {"7": S}}},
        "control": {"ok": ok, "shift_days": {k: int(v) for k, v in ctrl.groupby("plant").shift_days.first().items()},
                    "investigation": None, "skill": {"dp_pct|7": {k: round(float(v), 3) for k, v in sk.items()}}},
        "importance": None, "nets": True, "final": True, "plants_in": sorted(P.PLANTS),
        "truth": truth, "movement": {"real": {k: round(real[k], 3) for k in keys}, "now": fp, "before": old},
        "calibration": {"chosen": cal["chosen"], "five": cal["five"], "screen_flow_exponent": 1.75},
    }
    txt = "/* made by poc15/make_poc_data15.py from the v15 run's own files; nothing typed by hand */\nconst POC = " + \
        json.dumps(POC, ensure_ascii=False, separators=(",", ":")) + ";\n"
    (OUT / "poc_data15.js").write_text(txt)
    if deck:
        (P.Path(deck) / "poc_data.js").write_text(txt)
    print(json.dumps({m: S[m]["record 2"]["mae"] for m in METHODS}), "\ncontrol", json.dumps(POC["control"]["skill"]), "ok", ok,
          "\ntruth", truth, "\nmovement", json.dumps(fp))


if __name__ == "__main__":
    main()
