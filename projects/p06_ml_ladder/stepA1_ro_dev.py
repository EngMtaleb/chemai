"""Part A, development - will a train reach its review level within 28 days?

Everything here uses the 30 development plants of Project 3 (seed 1001, ten per
fouling law), normalised the way the real deposit is (vendor exponent 1.5). It
tunes and freezes every rung of the ladder; nothing here touches a test plant
or the real deposit's answers.

    python projects/p06_ml_ladder/stepA1_ro_dev.py
"""
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

from chemai.data.ro_sim import generate_plants
from chemai.features.ro_decisions import PHYSICS, PLAIN, sim_decisions
from chemai.models import deep
from chemai.models.ladder import classifiers, rp, threshold_at_precision, tune

HERE = Path(__file__).parent
OUT = HERE / "results"
SCALARS = ["level", "gap", "day", "now7", "cens42", "n42"]       # what the sequence networks also read
SLOW_CAP = {"svm_rbf": 4000}                                       # rows per fold for the slow model


def dev_table():
    f = OUT / "ro_dev.parquet"
    if f.exists():
        return pd.read_parquet(f), np.load(OUT / "ro_dev_seq.npy")
    df, seq = sim_decisions(generate_plants(30, seed=1001), list(range(30)))
    df.to_parquet(f)
    np.save(OUT / "ro_dev_seq.npy", seq)
    return df, seq


def dl_inputs(ev: pd.DataFrame, seq: np.ndarray, kind: str, st: deep.Standardiser):
    feats = {"mlp": PLAIN, "cnn": SCALARS, "gaplaw": SCALARS}[kind]
    X = st.transform(ev[feats].to_numpy(float))
    if kind == "mlp":
        return [X]
    S = deep.sequence_channels(seq)
    if kind == "cnn":
        return [X, S]
    now = ev["now7"].fillna(ev["last"]).to_numpy(np.float32)
    return [X, S, np.stack([ev["level"].to_numpy(np.float32), now], 1)]


def dl_model(kind: str, n_in: int):
    return {"mlp": lambda: deep.MLP(n_in), "cnn": lambda: deep.SeqCNN(3, n_in),
            "gaplaw": lambda: deep.GapLawNet(3, n_in)}[kind]()


def dl_fit(kind, ev, seq, y, g, idx, seed):
    """Train on rows `idx`, early-stopping on a fifth of their plants."""
    feats = {"mlp": PLAIN, "cnn": SCALARS, "gaplaw": SCALARS}[kind]
    gs = np.unique(g[idx])
    va_g = np.random.default_rng(seed).choice(gs, max(1, len(gs) // 5), replace=False)
    va, tr = idx[np.isin(g[idx], va_g)], idx[~np.isin(g[idx], va_g)]
    st = deep.Standardiser().fit(ev[feats].to_numpy(float)[tr])
    ins = dl_inputs(ev, seq, kind, st)
    m = deep.train(dl_model(kind, ins[0].shape[1]), [a[tr] for a in ins], y[tr], [a[va] for a in ins], y[va],
                   loss="bce_prob" if kind == "gaplaw" else "bce_logit", seed=seed)
    return m, st


def dl_predict(kind, m, st, ev, seq, idx):
    ins = dl_inputs(ev.iloc[idx].reset_index(drop=True), seq[idx], kind, st)
    return deep.predict(m, ins, prob=(kind != "gaplaw"))


def main():
    t0 = time.time()
    df, seq_all = dev_table()
    keep = (df["status"] != "insufficient").to_numpy()
    ev, seq = df[keep].reset_index(drop=True), seq_all[keep]
    y, g = ev["y"].to_numpy().astype(int), ev["plant"].to_numpy()
    res = {"data": dict(plants=30, seed=1001, normalisation="vendor 1.5", rows=int(len(ev)),
                        positives=int(y.sum()), cycles=int(ev["cycle"].nunique()),
                        dropped_insufficient=int((~keep).sum()))}
    base = {k: rp(y, ev[k].to_numpy()) for k in ("forecast_flag", "margin_flag")}
    base["union_post_hoc"] = rp(y, (ev["forecast_flag"] | ev["margin_flag"]).to_numpy())
    res["baselines"] = base
    target = base["margin_flag"]["precision"]
    res["threshold_rule"] = f"the threshold at which a model's development precision equals the margin rule's ({target})"
    oof = pd.DataFrame({"plant": g, "y": y})
    rungs = {}
    for fs_name, feats in (("ml", PLAIN), ("ml_physics", PLAIN + PHYSICS)):
        X = ev[feats]
        scores = {}
        for name, (model, grid) in classifiers().items():
            t = tune(name, model, grid, X, y, g, "cls", max_rows=SLOW_CAP.get(name))
            thr = threshold_at_precision(y, t.oof, target)
            scores[name] = dict(ap=round(t.cv_score, 4), params=t.params, threshold=thr,
                                at_threshold=rp(y, t.oof >= thr), seconds=round(t.seconds))
            oof[f"{fs_name}:{name}"] = t.oof
            print(fs_name, name, scores[name]["ap"], scores[name]["at_threshold"], flush=True)
        top3 = sorted(scores, key=lambda k: -scores[k]["ap"])[:3]
        ens = oof[[f"{fs_name}:{k}" for k in top3]].mean(1).to_numpy()
        thr = threshold_at_precision(y, ens, target)
        scores["ensemble_top3"] = dict(members=top3, ap=round(float(average_precision_score(y, ens)), 4),
                                       threshold=thr, at_threshold=rp(y, ens >= thr))
        oof[f"{fs_name}:ensemble_top3"] = ens
        rungs[fs_name] = scores
    for kind in ("mlp", "cnn", "gaplaw"):
        p = np.full(len(y), np.nan)
        for k, (tr, te) in enumerate(GroupKFold(5).split(ev, y, g)):
            m, st = dl_fit(kind, ev, seq, y.astype(np.float32), g, tr, seed=k)
            p[te] = dl_predict(kind, m, st, ev, seq, te)
        thr = threshold_at_precision(y, p, target)
        rungs.setdefault("dl", {})[kind] = dict(ap=round(float(average_precision_score(y, p)), 4), threshold=thr,
                                                at_threshold=rp(y, p >= thr))
        oof[f"dl:{kind}"] = p
        print("dl", kind, rungs["dl"][kind]["ap"], rungs["dl"][kind]["at_threshold"], flush=True)
    res["rungs"] = rungs
    # the model each rung sends to the test, chosen here by development AP
    pick = lambda d, ks: max(ks, key=lambda k: d[k]["ap"])          # noqa: E731
    seven = list(classifiers())
    res["frozen"] = {
        "ml": pick(rungs["ml"], seven), "ml_ensemble": "ensemble_top3",
        "ml_physics": pick(rungs["ml_physics"], seven), "ml_physics_ensemble": "ensemble_top3",
        "dl": pick(rungs["dl"], ["mlp", "cnn"]), "dl_physics": "gaplaw"}
    res["seconds"] = round(time.time() - t0)
    oof.to_parquet(OUT / "stepA1_dev_oof.parquet")
    (OUT / "stepA1_ro_dev.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res["frozen"]), res["seconds"], "s")


if __name__ == "__main__":
    main()
