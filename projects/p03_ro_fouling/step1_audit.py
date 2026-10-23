"""Step 1 - read the real deposit before modelling anything.

Downloads the Zenodo deposit (or uses data/ro_deposit/ if present), checks the
published md5, and records what the files are - and what they are not.

    python projects/p03_ro_fouling/step1_audit.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from chemai.config import RODepositConfig  # noqa: E402
from chemai.data.ro_deposit import (audit, cleaning_legs, download_deposit, group_campaigns,  # noqa: E402
                                    load_daily, load_ledger, recovery_by_campaign, split_cycles)

HERE = Path(__file__).parent
OUT, FIG = HERE / "results", HERE / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
COLOURS = {"A": "#2a6fb0", "B": "#128c84", "C": "#a8782f"}


def main():
    cfg = RODepositConfig()
    folder = download_deposit(cfg=cfg)
    daily, ledger = load_daily(folder / cfg.daily_file), load_ledger(folder / cfg.ledger_file)
    a = audit(daily, ledger, cfg)

    legs = cleaning_legs(daily)
    camps = {t: group_campaigns(v, cfg.campaign_gap_days) for t, v in legs.items()}
    cycles = split_cycles(daily, cfg)
    from scipy import stats
    rows = []
    for c in cycles:
        r = dict(cycle=c.id, train=c.train, kind=c.kind, start=c.start, end=c.end,
                 length=c.length, days_recorded=c.n)
        if c.n >= 8:
            ts = stats.theilslopes(c.dp_norm, c.day)
            r.update(dp_norm_first=round(float(np.median(c.dp_norm[:5])), 3),
                     dp_norm_last=round(float(np.median(c.dp_norm[-5:])), 3),
                     slope_per_week=round(float(ts[0] * 7), 4))
        rows.append(r)
    cyc = pd.DataFrame(rows)
    rec = recovery_by_campaign(daily, ledger)
    # DP_norm at each campaign (median of the 7 days before it): which cleanings had a hydraulic reason?
    at_clean = []
    for _, r in ledger.iterrows():
        g = daily[(daily["train"] == r["train"]) & daily["day_index"].between(r["day_index"] - 7, r["day_index"] - 1)]
        at_clean.append(round(float(g["dp_norm"].median()), 3) if len(g) else None)
    ledger = ledger.assign(dp_norm_before=at_clean)

    res = dict(audit=a, cycles=cyc.to_dict("records"), recovery=rec.round(4).to_dict("records"),
               ledger=ledger.to_dict("records"),
               campaigns={t: [list(x) for x in v] for t, v in camps.items()},
               complete_cycles=int((cyc["kind"] == "complete").sum()),
               citation=cfg.citation)
    (OUT / "step1_audit.json").write_text(json.dumps(res, indent=1, default=str))

    # ---- figure: the deposit, one panel per train
    fig, axes = plt.subplots(3, 1, figsize=(13, 8.5), sharex=True)
    for ax, (t, g) in zip(axes, daily.groupby("train")):
        ok = ~g["censored"]
        ax.plot(g.loc[ok, "day_index"], g.loc[ok, "dp_norm"], ".", ms=3.5, color=COLOURS[t],
                label="DP_norm (1 + dp_pct)")
        ax.plot(g.loc[~ok, "day_index"], g.loc[~ok, "dp_norm"], "|", ms=6, color="#9db3c0",
                label="zero = censored at the baseline")
        for s, e in camps[t]:
            ax.axvspan(s - 0.5, e + 0.5, color="#b53c58", alpha=0.35, lw=0)
        ax.axhline(1.15, color="#b53c58", ls="--", lw=1)
        ax.set_ylabel(f"train {t}")
        ax.set_ylim(0.95, 1.6)
        ax.grid(alpha=0.25)
    axes[0].legend(loc="upper left", fontsize=9, frameon=False)
    axes[0].set_title("Public deposit: normalised feed-channel pressure drop, three trains, "
                      "18 cleaning campaigns (red)", fontsize=11, loc="left")
    axes[-1].set_xlabel("day (calendar dates withheld by the authors)")
    fig.tight_layout()
    fig.savefig(FIG / "step1_deposit.png", dpi=110)

    # ---- figure: does the indicator reset at a cleaning?
    post_dp, post_ndp = [], []
    for t, L in legs.items():
        g = daily[daily["train"] == t].set_index("day_index")
        for c in L:
            w = g[(g.index > c) & (g.index <= c + 14)]
            post_dp += list(w["dp_pct"])
            post_ndp += list(w["ndp_pct"])
    fig, ax = plt.subplots(figsize=(7.5, 4))
    bins = np.linspace(0, 0.5, 26)
    ax.hist(post_ndp, bins=bins, alpha=0.75, color="#c4791b", label=f"ndp_pct  median {np.median(post_ndp):.3f}")
    ax.hist(post_dp, bins=bins, alpha=0.75, color="#2a6fb0", label=f"dp_pct  median {np.median(post_dp):.3f}")
    ax.set_xlabel("value in the 14 days after a cleaning")
    ax.set_ylabel("train-days")
    ax.set_title("ndp_pct falls to zero after every cleaning (reset); dp_pct does not", fontsize=10, loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "step1_reset_check.png", dpi=110)

    print(json.dumps(a, indent=1))
    print(cyc.to_string())
    print(ledger.to_string())


if __name__ == "__main__":
    main()
