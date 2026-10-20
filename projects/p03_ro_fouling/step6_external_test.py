"""Step 6 - the external test: the frozen method, once, on the real deposit.

Nothing below was tuned on these data. The settings are those frozen in step 4
and checked on the test plants in step 5 (ROConfig defaults). Declared before
the run: primary = linear law; secondary = the gap law, reported, not adopted.

    python projects/p03_ro_fouling/step6_external_test.py
"""
from dataclasses import replace
import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from chemai.config import RODepositConfig, ROConfig  # noqa: E402
from chemai.data.ro_deposit import download_deposit, load_daily, load_ledger, split_cycles  # noqa: E402
from chemai.evaluation.cip_plan import TrainRecord, build_plan  # noqa: E402
from chemai.evaluation.fouling_eval import real_decisions, score_real  # noqa: E402
from chemai.models.fouling_forecast import forecast_cycle, prepare_window  # noqa: E402

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
OUT, FIG = HERE / "results", HERE / "figures"
LEVELS = (1.15, 1.20)


def crossing_summary(cr: pd.DataFrame) -> dict:
    g = cr[(cr["truth"] > 0) & (cr["status"] != "insufficient")]
    unc = g[~g["truth_censored"] & (g["truth"] <= 180)]
    dated = unc[unc["status"] == "crossing"]
    err = (dated["p50"] - dated["truth"]).abs()
    lo, hi = unc["p10"].fillna(1e9), unc["p90"].fillna(np.inf)
    covered = (unc["truth"] >= lo.where(unc["status"] == "crossing", 0)) & (unc["truth"] <= hi)
    cen = g[g["truth_censored"]]
    # a censored truth says "not before the cleaning": a forecast agrees when its P50 lies
    # beyond the cleaning, or when it saw no crossing at all
    agree = (cen["p50"].isna()) | (cen["p50"] >= cen["truth"])
    out = dict(decisions=int(len(g)), cycles=int(g["cycle"].nunique()),
               observed_crossings=int(len(unc)), dated_of_those=int(len(dated)),
               median_abs_err_days=round(float(err.median()), 1) if len(err) else None,
               mae_days=round(float(err.mean()), 1) if len(err) else None,
               coverage_uncensored=round(float(covered.mean()), 3) if len(unc) else None,
               censored=int(len(cen)), censored_consistent=round(float(agree.mean()), 3) if len(cen) else None)
    fl, ok = dated[~dated["in_envelope"]], dated[dated["in_envelope"]]
    if len(fl) and len(ok):
        out.update(flagged=int(len(fl)),
                   median_err_inside=round(float((ok["p50"] - ok["truth"]).abs().median()), 1),
                   median_err_flagged=round(float((fl["p50"] - fl["truth"]).abs().median()), 1))
    return out


def level_summary(lv: pd.DataFrame) -> dict:
    out = {}
    for h, g in lv.groupby("horizon"):
        ef = (g["predicted"] - g["measured"]).abs()
        ep = (g["persistence"] - g["measured"]).abs()
        # cycle bootstrap of the difference: a cycle, not a day, is the independent unit
        rng = np.random.default_rng(0)
        cyc = g["cycle"].unique()
        diffs = []
        for _ in range(2000):
            pick = rng.choice(cyc, len(cyc), replace=True)
            gg = pd.concat([g[g["cycle"] == c] for c in pick])
            diffs.append(float((gg["predicted"] - gg["measured"]).abs().mean()
                               - (gg["persistence"] - gg["measured"]).abs().mean()))
        out[int(h)] = dict(n=int(len(g)), cycles=int(len(cyc)), mae_forecast=round(float(ef.mean()), 4),
                           mae_persistence=round(float(ep.mean()), 4),
                           difference_ci90=[round(float(np.quantile(diffs, 0.05)), 4),
                                            round(float(np.quantile(diffs, 0.95)), 4)],
                           forecast_better_share=round(float((ef < ep).mean()), 3))
    return out


def decision_curves(cr: pd.DataFrame) -> dict:
    g = real_decisions(cr)
    y = g["y"].to_numpy().astype(bool)
    p50 = g["p50"].fillna(np.inf).to_numpy()
    crossing = (g["status"] == "crossing").to_numpy()

    def pr(flag):
        tp, fp, fn = (flag & y).sum(), (flag & ~y).sum(), (~flag & y).sum()
        return [int(tp), int(fp), int(fn), round(tp / max(tp + fn, 1), 3), round(tp / max(tp + fp, 1), 3)]
    obs, lvl = g["obs_level_now"].to_numpy(), g["level"].to_numpy()
    fc = crossing & (p50 <= 28)
    return dict(n=int(len(g)), positives=int(y.sum()), cycles=int(g["cycle"].nunique()),
                forecast_p50={k: pr(crossing & (p50 <= k)) for k in (14, 28, 42)},
                margin={d: pr(obs >= lvl - d) for d in (0.01, 0.02, 0.03, 0.05, 0.08)},
                post_hoc_union_0_03=pr(fc | (obs >= lvl - 0.03)))


def cycle_cleaning_info(daily: pd.DataFrame, cycles) -> dict:
    """pre / post / previous-post DP_norm (7-day medians) for the cleaning that opened each cycle."""
    info, prev_post = {}, {}
    for c in cycles:
        g = daily[daily["train"] == c.train].set_index("day_index")["dp_norm"]
        if c.kind == "initial":
            prev_post[c.train] = float(g[g.index <= c.start + 14].median())
            continue
        pre = g[(g.index < c.start) & (g.index >= c.start - 10)]
        post = g[(g.index > c.start) & (g.index <= c.start + 10)]
        info[c.id] = dict(pre=float(pre.median()) if len(pre) else None,
                          post=float(post.median()) if len(post) else None,
                          previous_post=prev_post.get(c.train))
        if len(post):
            prev_post[c.train] = float(post.median())
    return info


def plan_replay(daily: pd.DataFrame, ledger: pd.DataFrame, cycles, cfg: ROConfig) -> pd.DataFrame:
    """The weekly plan as it would have read every 7 days, per train - with the plan
    rules as finally adopted (the margin trigger was added after this test)."""
    info = cycle_cleaning_info(daily, cycles)
    rows = []
    last_day = int(daily["day_index"].max())
    for day in range(14, last_day + 1, 7):
        trains = []
        for t in sorted(daily["train"].unique()):
            cur = [c for c in cycles if c.train == t and c.start <= day < c.end]
            if not cur:
                continue                                  # inside a cleaning
            c = cur[0]
            m = c.abs_day <= day
            k = info.get(c.id, {})
            trains.append(TrainRecord(name=t, day=list(c.day[m]), dp_norm=list(c.dp_norm[m]),
                                      censored=list(c.censored[m]), pre_cleaning=k.get("pre"),
                                      post_cleaning=k.get("post"), previous_post_cleaning=k.get("previous_post")))
        if not trains:
            continue
        plan = build_plan(trains, cfg, trains_total=3)
        for r in plan.rows:
            rows.append(dict(day=day, train=r.train, category=r.category, week=r.week,
                             level_now=r.level_now, p50=r.days_p50, flags=len(r.flags)))
    return pd.DataFrame(rows)


def main():
    dcfg, cfg = RODepositConfig(), ROConfig()
    folder = download_deposit(cfg=dcfg)
    daily, ledger = load_daily(folder / dcfg.daily_file), load_ledger(folder / dcfg.ledger_file)
    cycles = split_cycles(daily, dcfg)

    res = {"frozen": dict(law=cfg.law, window=cfg.fit_window_days, inflation=cfg.interval_inflation,
                          ratio_max=cfg.extrapolation_ratio_max, plan_days=cfg.plan_weeks * 7),
           "data": dcfg.citation}
    cr, lv = score_real(cycles, LEVELS, cfg.law, cfg)
    cr.to_csv(OUT / "step6_crossings.csv", index=False)
    res["primary"] = dict(crossing={str(lv_): crossing_summary(cr[cr["level"] == lv_]) for lv_ in LEVELS},
                          crossing_all=crossing_summary(cr), level=level_summary(lv),
                          decisions=decision_curves(cr))
    cfg_b = replace(cfg, law="blockage")
    crb, lvb = score_real(cycles, LEVELS, "blockage", cfg_b)
    res["secondary_blockage"] = dict(crossing_all=crossing_summary(crb), level=level_summary(lvb))
    print(json.dumps(res["primary"]["crossing_all"]), "\n", json.dumps(res["secondary_blockage"]["crossing_all"]))
    print(json.dumps(res["primary"]["level"]), "\n", json.dumps(res["secondary_blockage"]["level"]))
    print(json.dumps(res["primary"]["decisions"]))

    # ---- the weekly plan, replayed, against what the operators did
    rp = plan_replay(daily, ledger, cycles, cfg)
    rp.to_csv(OUT / "step6_plan_replay.csv", index=False)
    before = []
    for _, r in ledger.iterrows():
        w = rp[(rp["train"] == r["train"]) & (rp["day"] < r["day_index"]) & (rp["day"] >= r["day_index"] - 7)]
        before.append(dict(episode=r["episode"], day=int(r["day_index"]),
                           plan_week_before=w["category"].iloc[-1] if len(w) else "no plan",
                           level_now=float(w["level_now"].iloc[-1]) if len(w) and pd.notna(w["level_now"].iloc[-1]) else None))
    res["plan_before_each_cleaning"] = before
    res["plan_category_share"] = rp["category"].value_counts(normalize=True).round(3).to_dict()
    # the plan on the last day of the record
    end = int(daily["day_index"].max())
    last = rp[rp["day"] == rp["day"].max()]
    res["plan_at_end"] = dict(day=int(rp["day"].max()), rows=last.to_dict("records"))
    print(pd.DataFrame(before).to_string())
    (OUT / "step6_external_test.json").write_text(json.dumps(res, indent=1, default=str))

    # ---- figure: four real cycles, one forecast each
    picks = [("C:345-481", 70), ("A:579-727", 70), ("C:586-725", 70), ("B:545-584", 28)]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5))
    for ax, (cid, d) in zip(axes.ravel(), picks):
        c = next(x for x in cycles if x.id == cid)
        m = c.day <= d
        f = forecast_cycle(c.day[m], c.dp_norm[m], 1.20, censored=c.censored[m], cfg=cfg)
        tw, _, _ = prepare_window(c.day[m], c.dp_norm[m], cfg, c.censored[m])
        ax.axvspan(tw[0], tw[-1], color="#e8f4f6", lw=0)
        ax.plot(c.day[m], c.dp_norm[m], ".", color="#2a6fb0", ms=4, label="seen at the decision")
        ax.plot(c.day[~m], c.dp_norm[~m], ".", color="#9db3c0", ms=4, label="what came next")
        ax.axhline(1.20, color="#b53c58", ls="--", lw=1.2)
        ax.axvline(d, color="#14364c", lw=1)
        if f.status == "crossing":
            x1 = d + f.days_to_level
            ax.plot([d, x1], [f.level_now, 1.20], color="#128c84", lw=2.2, label="forecast")
            ax.axvspan(d + f.days_lo, d + (f.days_hi if f.days_hi is not None else 180), ymin=0.0, ymax=0.05,
                       color="#128c84", alpha=0.6)
        title = f"train {c.train}, days {c.start}-{c.end}: {f.status}"
        if f.status == "crossing":
            title += f", {f.days_to_level:.0f} d (P10 {f.days_lo:.0f}, P90 " + \
                     (f"{f.days_hi:.0f})" if f.days_hi is not None else ">180)")
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel("days since the cleaning")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("The frozen forecast on four real cycles (review level 1.20)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "step6_cases.png", dpi=110)

    # ---- figure: the plan replayed against the operators' cleanings
    cat_col = {"review": "#b53c58", "plan": "#e3a33b", "watch": "#9fc6d6", "none": "#cfe3dc",
               "wait": "#eef2f4", "data": "#d5d5d5", "limit": "#6d1b2f"}
    fig, ax = plt.subplots(figsize=(13, 3.6))
    for i, t in enumerate(["A", "B", "C"]):
        g = rp[rp["train"] == t]
        for _, r in g.iterrows():
            ax.barh(i, 7, left=r["day"] - 7, color=cat_col[r["category"]], height=0.6, lw=0)
        for d in ledger.loc[ledger["train"] == t, "day_index"]:
            ax.plot([d, d], [i - 0.38, i + 0.38], color="#14364c", lw=2.2)
    ax.set_yticks([0, 1, 2], ["train A", "train B", "train C"])
    ax.set_xlabel("day   (black ticks: the operators' cleanings)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=v) for v in cat_col.values()]
    ax.legend(handles, list(cat_col), ncol=7, frameon=False, fontsize=8.5, loc="upper center",
              bbox_to_anchor=(0.5, 1.22))
    fig.tight_layout()
    fig.savefig(FIG / "step6_plan_replay.png", dpi=110)


if __name__ == "__main__":
    main()
