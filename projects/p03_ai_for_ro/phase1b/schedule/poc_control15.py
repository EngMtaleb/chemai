"""declaration_v15.md, the negative control only (poc_trees15.py's second half, unchanged; the main run's forecasts are in
out/pred_trees15.csv): XGBoost without SDI and turbidity, and with them shifted (yearly cycle out, rest moved, cycle back)."""
import time
import pandas as pd
import poc_lib15 as Q
import poc_trees15 as T
P = Q.P


def main():
    t, c = Q.load()
    days, feats, pieces = Q.merged(t, c)
    minus = [f for f in feats if f not in P.extra_feats(P.INSTRUMENTS)]
    ctrl = [T.run(days, minus, "record 2 without SDI and turbidity")]
    days_s, feats_s, pieces_s = Q.merged(t, c, shifted=True)
    assert (days_s.valid == days.valid).all() and (days_s.t == days.t).all() and feats_s == feats
    ctrl.append(T.run(days_s, feats, "record 2, SDI and turbidity shifted"))
    pd.concat(ctrl, ignore_index=True).assign(shift_days=lambda x: x.plant.map({p: v[2] for p, v in pieces_s.items()})) \
        .to_csv(T.OUT / "pred_control15.csv", index=False)
    print(f"control done [{time.time() - T.T0:.0f} s]", flush=True)


if __name__ == "__main__":
    main()
