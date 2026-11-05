"""declaration_v15.md: no change, the fouling law, XGBoost and the hybrid on the merged dataset (record 2, dp_pct, 7 days,
rolling); the negative control (XGBoost without SDI and turbidity; with them shifted). Every forecast saved."""
import time
import numpy as np, pandas as pd
import poc_lib15 as Q
P, L = Q.P, Q.L
OUT = Q.HERE / "out"
T0 = time.time()


def run(days, feats, label):
    pairs = days[days.valid].reset_index(drop=True)
    out = []
    for b, train, te in Q.folds(pairs):
        (f,) = L.forecasts(train, [te], Q.C, Q.H, feats)
        base = te[["train", "t", Q.C, f"y_{Q.C}"]].rename(columns={Q.C: "now", f"y_{Q.C}": "y"})
        out.append(base.assign(plant=te.train.str[:2], record=label, block=b, **{m: f[m] for m in Q.METHODS}))
        print(f"{label} block {b}: train {len(train)}, test {len(te)} [{time.time() - T0:.0f} s]", flush=True)
    return pd.concat(out, ignore_index=True)


def main():
    t, c = Q.load()
    days, feats, pieces = Q.merged(t, c)
    print(f"merged: {int((days.valid & (days.h == Q.H)).sum())} pairs 7 days ahead, {len(feats)} inputs [{time.time() - T0:.0f} s]", flush=True)
    run(days, feats, "record 2").to_csv(OUT / "pred_trees15.csv", index=False)
    minus = [f for f in feats if f not in P.extra_feats(P.INSTRUMENTS)]
    ctrl = [run(days, minus, "record 2 without SDI and turbidity")]
    days_s, feats_s, pieces_s = Q.merged(t, c, shifted=True)
    assert (days_s.valid == days.valid).all() and (days_s.t == days.t).all() and feats_s == feats
    ctrl.append(run(days_s, feats, "record 2, SDI and turbidity shifted"))
    pd.concat(ctrl, ignore_index=True).assign(shift_days=lambda x: x.plant.map({p: v[2] for p, v in pieces_s.items()})) \
        .to_csv(OUT / "pred_control15.csv", index=False)
    print(f"trees done [{time.time() - T0:.0f} s]", flush=True)


if __name__ == "__main__":
    main()
