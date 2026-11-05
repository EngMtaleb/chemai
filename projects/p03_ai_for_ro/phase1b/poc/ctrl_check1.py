"""Control investigation, step 1 (no refits): is the shifted build identical to the as-run build everywhere except the
twelve SDI and turbidity inputs? And how seasonal are SDI and turbidity (their yearly cycle survives a circular shift)?"""
import numpy as np, pandas as pd
import poc_lib as P
t, c = P.load()
inst = set(P.extra_feats(P.INSTRUMENTS))
for p in P.PLANTS:
    d, led, _ = P.plant(t, c, p)
    ds, leds, k = P.plant(t, c, p, shift_instruments=True)
    a, b = P.build_days(d, led), P.build_days(ds, leds)
    same_shape = a.shape == b.shape and list(a.columns) == list(b.columns)
    other = [col for col in a.columns if col not in inst]
    diff = [col for col in other if not a[col].equals(b[col])]
    moved = [col for col in inst if not np.allclose(a[col].fillna(-9), b[col].fillna(-9))]
    print(p, "k", k, "shape same", same_shape, "| other columns that differ:", diff, "| instrument inputs moved:", len(moved), "of", len(inst))
# yearly cycle in the plant-wide instruments: share of variance explained by a 365-day sine + cosine
pw = pd.read_csv(P.DATA / "plantwide.csv")
for p, g in pw.groupby("plant"):
    g = g.sort_values("day"); x = np.arange(len(g))
    X = np.column_stack([np.ones(len(g)), np.sin(2*np.pi*x/365), np.cos(2*np.pi*x/365)])
    for col in ("sdi", "turbidity_ntu"):
        y = np.log(g[col].to_numpy())
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        r2 = 1 - np.var(y - X @ beta) / np.var(y)
        print(p, col, "yearly-cycle share of variance (log scale):", round(r2, 3))
