"""Part A, the test - run once, after DECLARED_BEFORE_RUN_A.md is committed.

1. Final exam: 60 NEW simulated plants (seed 6006), made by this script after the
   declaration; every rung frozen in step A1 is trained on all development plants
   and scored here once.
2. The real deposit, exploratory: its answers were seen in Project 3's external
   test, so no pass mark is attached; the frozen models and thresholds run as they are.
3. Exploratory: the same ladder trained on 30 plants of the quieter "matched"
   simulator (settings chosen after Project 3 saw the deposit), scored on the deposit.

    python projects/p06_ml_ladder/stepA2_ro_test.py
"""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from chemai.config import RODepositConfig, ROSimConfig
from chemai.data.ro_deposit import download_deposit, load_daily, split_cycles
from chemai.data.ro_sim import generate_plants
from chemai.features.ro_decisions import PHYSICS, PLAIN, real_decisions_table, sim_decisions
from chemai.models.ladder import classifiers, fit_final, group_bootstrap, rp

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from stepA1_ro_dev import SLOW_CAP, dl_fit, dl_predict  # noqa: E402

OUT = HERE / "results"
DECL = HERE / "DECLARED_BEFORE_RUN_A.md"
MATCHED = dict(dp_noise_rel=(0.004, 0.012), outlier_probability=0.02, operator_level=(1.10, 1.25),
               max_days_without_cip=(45, 120), min_days_between_cip=14, other_reason_cip_per_year=2.0)


def declaration_commit() -> str:
    """The commit that holds the declaration; refuse to run if it is missing or edited."""
    root = HERE.parents[1]
    rel = str(DECL.relative_to(root))
    dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--", rel],
                           capture_output=True, text=True).stdout.strip()
    sha = subprocess.run(["git", "-C", str(root), "log", "-1", "--format=%h", "--", rel],
                         capture_output=True, text=True).stdout.strip()
    if not DECL.exists() or dirty or not sha:
        raise SystemExit(f"{rel} must be committed and unchanged before the test runs")
    return sha


def fit_all(dev: pd.DataFrame, dseq: np.ndarray, frozen: dict) -> dict:
    """Every rung, trained on all development rows with its frozen settings."""
    y, g = dev["y"].to_numpy().astype(int), dev["plant"].to_numpy()
    models = {}
    for fs_name, feats in (("ml", PLAIN), ("ml_physics", PLAIN + PHYSICS)):
        for name, (model, _) in classifiers().items():
            p = frozen["rungs"][fs_name][name]["params"]
            models[f"{fs_name}:{name}"] = ("sk", feats, fit_final(model, p, dev[feats], y, SLOW_CAP.get(name)))
    idx = np.arange(len(dev))
    for kind in ("mlp", "cnn", "gaplaw"):
        m, st = dl_fit(kind, dev, dseq, y.astype(np.float32), g, idx, seed=0)
        models[f"dl:{kind}"] = ("dl", kind, (m, st))
    return models


def predict_all(models: dict, frozen: dict, ev: pd.DataFrame, seq: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame(index=ev.index)
    for key, (kind, feats, m) in models.items():
        if kind == "sk":
            out[key] = m.predict_proba(ev[feats])[:, 1]
        else:
            out[key] = dl_predict(feats, m[0], m[1], ev, seq, np.arange(len(ev)))
    for fs_name in ("ml", "ml_physics"):
        mem = frozen["rungs"][fs_name]["ensemble_top3"]["members"]
        out[f"{fs_name}:ensemble_top3"] = out[[f"{fs_name}:{k}" for k in mem]].mean(1)
    return out


def threshold_of(frozen: dict, key: str) -> float:
    rung, name = key.split(":")
    return frozen["rungs"][rung][name]["threshold"]


def score(ev: pd.DataFrame, pred: pd.DataFrame, frozen: dict, groups: np.ndarray) -> dict:
    y = ev["y"].to_numpy().astype(bool)
    margin = ev["margin_flag"].to_numpy()
    res = {"rows": int(len(ev)), "positives": int(y.sum()), "groups": int(len(np.unique(groups))),
           "forecast": rp(y, ev["forecast_flag"].to_numpy()), "margin_rule": rp(y, margin),
           "union_post_hoc": rp(y, (ev["forecast_flag"] | ev["margin_flag"]).to_numpy()), "models": {}}
    r_m = res["margin_rule"]["recall"]
    for key in pred.columns:
        p = pred[key].to_numpy()
        flag = p >= threshold_of(frozen, key)
        d = rp(y, flag)
        d["ap"] = round(float(average_precision_score(y, p)), 3)

        def diff(i, f=flag):
            yy = y[i]
            if yy.sum() == 0:
                return np.nan
            return (f[i] & yy).sum() / yy.sum() - (margin[i] & yy).sum() / yy.sum()
        lo, hi = group_bootstrap(groups, diff)
        d["recall_minus_rule"] = round(d["recall"] - r_m, 3)
        d["recall_minus_rule_ci95"] = [round(lo, 3), round(hi, 3)]
        res["models"][key] = d
    return res


def main():
    t0 = time.time()
    sha = declaration_commit()
    frozen = json.loads((OUT / "stepA1_ro_dev.json").read_text())
    dev_all = pd.read_parquet(OUT / "ro_dev.parquet")
    dseq_all = np.load(OUT / "ro_dev_seq.npy")
    keep = (dev_all["status"] != "insufficient").to_numpy()
    dev, dseq = dev_all[keep].reset_index(drop=True), dseq_all[keep]
    res = {"declaration_commit": sha, "frozen_picks": frozen["frozen"]}

    models = fit_all(dev, dseq, frozen)
    # 1. the final exam
    test, tseq = sim_decisions(generate_plants(60, seed=6006), list(range(100, 160)))
    k = (test["status"] != "insufficient").to_numpy()
    te, ts = test[k].reset_index(drop=True), tseq[k]
    pt = predict_all(models, frozen, te, ts)
    res["final_exam"] = dict(plants=60, seed=6006, **score(te, pt, frozen, te["plant"].to_numpy()))
    pd.concat([te[["plant", "cycle", "law_true", "level", "decision_day", "y", "forecast_flag", "margin_flag"]], pt],
              axis=1).to_parquet(OUT / "stepA2_final_exam_predictions.parquet")
    print(json.dumps({k: v for k, v in res["final_exam"].items() if k != "models"}), flush=True)

    # 2. the real deposit, exploratory
    dcfg = RODepositConfig()
    cycles = split_cycles(load_daily(download_deposit(cfg=dcfg) / dcfg.daily_file), dcfg)
    real, rseq = real_decisions_table(cycles)
    k = (real["status"] != "insufficient").to_numpy()
    re_, rs = real[k].reset_index(drop=True), rseq[k]
    pr_ = predict_all(models, frozen, re_, rs)
    res["real_exploratory"] = score(re_, pr_, frozen, re_["cycle"].to_numpy())
    pd.concat([re_[["train", "cycle", "level", "decision_day", "y", "forecast_flag", "margin_flag"]], pr_],
              axis=1).to_parquet(OUT / "stepA2_real_predictions.parquet")

    # 3. exploratory: trained on the matched simulator instead
    mdev, mseq = sim_decisions(generate_plants(30, seed=1005, cfg=replace(ROSimConfig(), **MATCHED)), list(range(30)))
    k = (mdev["status"] != "insufficient").to_numpy()
    mdev, mseq = mdev[k].reset_index(drop=True), mseq[k]
    from chemai.models.ladder import threshold_at_precision
    from sklearn.model_selection import GroupKFold
    from chemai.models.ladder import tune
    my, mg = mdev["y"].to_numpy().astype(int), mdev["plant"].to_numpy()
    target = rp(my, mdev["margin_flag"].to_numpy())["precision"]
    mres = {"plants": 30, "seed": 1005, "rows": int(len(mdev)), "margin_rule_dev": rp(my, mdev["margin_flag"].to_numpy()),
            "models": {}}
    yreal = re_["y"].to_numpy().astype(bool)
    for fs_name, feats in (("ml", PLAIN), ("ml_physics", PLAIN + PHYSICS)):
        name = frozen["frozen"][fs_name]
        model, grid = classifiers()[name]
        t = tune(name, model, grid, mdev[feats], my, mg, "cls")
        thr = threshold_at_precision(my, t.oof, target)
        m = fit_final(model, t.params, mdev[feats], my)
        p = m.predict_proba(re_[feats])[:, 1]
        d = rp(yreal, p >= thr)
        d["ap"] = round(float(average_precision_score(yreal, p)), 3)
        mres["models"][f"{fs_name}:{name}"] = d
    res["real_exploratory_matched_training"] = mres
    res["seconds"] = round(time.time() - t0)
    (OUT / "stepA2_ro_test.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res["real_exploratory"]["margin_rule"]), res["seconds"], "s")


if __name__ == "__main__":
    main()
