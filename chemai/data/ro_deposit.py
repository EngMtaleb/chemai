"""The public RO deposit - loader, cleaning cycles, and the audit that runs first.

Source: Zenodo record 21055948 (deposit v1.3, CC-BY-4.0), accompanying
Su, Meng and Chang (2026), Membranes 16(8), 257. Three seawater RO trains,
about two years of daily, train-level indicators derived from 10-min SCADA,
and an anonymised cleaning (CIP) ledger. Calendar dates are withheld: days are
integers counted from the first record. Not committed - `download_deposit`
fetches the zip and checks the md5 published on the record page.

What the files are, as found by reading them (projects/p03_ro_fouling/VALIDATION.md):

  * `dp_pct` is the normalised feed-channel pressure drop against a FIXED
    commissioning baseline, minus one and clipped to [0, 1]: DP_norm = 1 + dp_pct.
    It is NOT reset at a cleaning. The deposit's own build script calls all four
    indicators "per-CIP baseline"; the data contradict that for dp_pct (after a
    cleaning it stays at its post-cleaning level) and agree for ndp_pct (it
    falls to zero after every cleaning).
  * A zero means "at or below the clean baseline", not "clean": the value is
    censored there. 15 % of all train-days are zeros.
  * The table counts cleanings by LEG (23 acid/alkaline stages); the ledger
    counts campaigns (18). Legs closer than 7 days are one campaign.
  * Six plant days hold more than 144 ten-minute samples - impossible for a
    10-min log. Flagged, not repaired.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import io
import logging
from pathlib import Path
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from chemai.config import DATA_DIR, RODepositConfig

log = logging.getLogger(__name__)

DAILY_COLUMNS = ("train", "day_index", "n_10min_samples", "basis", "ndp_pct", "dp_pct",
                 "flux_pct", "sp_pct", "days_since_cip", "days_to_next_cip", "state")
LEDGER_COLUMNS = ("train", "episode", "day_index", "stage")


# ------------------------------------------------------------------ files

def deposit_dir(cfg: RODepositConfig | None = None) -> Path:
    cfg = cfg or RODepositConfig()
    return DATA_DIR / cfg.folder


def md5_of(path: Path | str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def download_deposit(dest: Path | str | None = None, cfg: RODepositConfig | None = None,
                     force: bool = False, timeout: float = 60.0) -> Path:
    """Fetch the Zenodo zip, check its md5, and extract the two data files.

    The md5 is the one printed on the record page. A file that does not match
    is refused rather than used: an analysis built on a silently different file
    cannot be reproduced by anyone else.
    """
    cfg = cfg or RODepositConfig()
    dest = Path(dest) if dest else deposit_dir(cfg)
    dest.mkdir(parents=True, exist_ok=True)
    if not force and (dest / cfg.daily_file).exists() and (dest / cfg.ledger_file).exists():
        return dest
    zpath = dest / cfg.zip_name
    if force or not zpath.exists():
        log.info("downloading %s", cfg.url)
        with urllib.request.urlopen(cfg.url, timeout=timeout) as r:   # noqa: S310 - fixed https URL
            zpath.write_bytes(r.read())
    got = md5_of(zpath)
    if got != cfg.md5:
        raise ValueError(f"{zpath.name}: md5 {got} does not match the published {cfg.md5}")
    extract_deposit(zpath, dest, cfg)
    return dest


def extract_deposit(zpath: Path | str, dest: Path | str, cfg: RODepositConfig | None = None) -> Path:
    """Extract the two data files (and the column dictionary) from the zip."""
    cfg = cfg or RODepositConfig()
    dest = Path(dest)
    with zipfile.ZipFile(zpath) as z:
        names = {Path(n).name: n for n in z.namelist()}
        for want in (cfg.daily_file, cfg.ledger_file, "column_dictionary.md", "README.md"):
            if want not in names:
                raise ValueError(f"{want} missing from {Path(zpath).name}")
            (dest / want).write_bytes(z.read(names[want]))
    return dest


def _path(name: str, path: Path | str | None, cfg: RODepositConfig) -> Path:
    p = Path(path) if path else deposit_dir(cfg) / name
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. The deposit is not committed - run "
            "chemai.data.ro_deposit.download_deposit() (Zenodo record 21055948).")
    return p


def load_daily(path: Path | str | None = None, cfg: RODepositConfig | None = None) -> pd.DataFrame:
    """The daily train-level table, with three columns added and nothing changed.

    dp_norm         1 + dp_pct (the fixed-baseline normalised pressure drop)
    censored        dp_pct == 0: at or below the clean baseline, value unknown
    over_max        more 10-min samples in the day than 10-min logging allows
    """
    cfg = cfg or RODepositConfig()
    df = pd.read_csv(_path(cfg.daily_file, path, cfg), encoding="utf-8-sig")
    missing = [c for c in DAILY_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"daily table is missing columns {missing}")
    df = df.sort_values(["train", "day_index"]).reset_index(drop=True)
    if df.duplicated(["train", "day_index"]).any():
        raise ValueError("more than one row for a train-day")
    df["dp_norm"] = 1.0 + df["dp_pct"]
    df["censored"] = df["dp_pct"] <= 0.0
    df["over_max"] = df["n_10min_samples"] > cfg.max_samples_per_day
    return df


def load_ledger(path: Path | str | None = None, cfg: RODepositConfig | None = None) -> pd.DataFrame:
    cfg = cfg or RODepositConfig()
    led = pd.read_csv(_path(cfg.ledger_file, path, cfg), encoding="utf-8-sig")
    missing = [c for c in LEDGER_COLUMNS if c not in led.columns]
    if missing:
        raise ValueError(f"ledger is missing columns {missing}")
    return led.sort_values(["train", "day_index"]).reset_index(drop=True)


# ---------------------------------------------------------------- cleanings

def cleaning_legs(daily: pd.DataFrame) -> dict[str, list[int]]:
    """The cleaning days the table itself was built on, per train.

    Recovered from days_since_cip and days_to_next_cip: every row points back
    to the previous cleaning and forward to the next. These are LEGS (acid and
    alkaline stages counted separately), not the ledger's campaigns.
    """
    out: dict[str, list[int]] = {}
    for t, g in daily.groupby("train"):
        back = (g["day_index"] - g["days_since_cip"]).dropna()
        fwd = (g["day_index"] + g["days_to_next_cip"]).dropna()
        out[str(t)] = sorted({int(round(v)) for v in pd.concat([back, fwd])})
    return out


def group_campaigns(legs: list[int], gap_days: int = 7) -> list[tuple[int, int]]:
    """Collapse legs closer than `gap_days` into (first_day, last_day) campaigns."""
    if not legs:
        return []
    legs = sorted(legs)
    out, start, last = [], legs[0], legs[0]
    for d in legs[1:]:
        if d - last <= gap_days:
            last = d
        else:
            out.append((start, last))
            start = last = d
    out.append((start, last))
    return out


@dataclass
class Cycle:
    """One stretch of operation between two cleanings, for one train.

    `day` is counted from the end of the cleaning that opened the cycle (or from
    the first record for the initial stretch). The cycle is the unit of every
    split and every score: days inside one cycle share a fouling layer, a
    cleaning and an operator's decision, so they are not independent samples.
    """
    train: str
    start: int                    # last leg of the opening campaign (or first record)
    end: int                      # first leg of the closing campaign (or last record + 1)
    kind: str                     # 'initial' | 'complete' | 'open'
    day: np.ndarray = field(repr=False)        # days since start
    dp_norm: np.ndarray = field(repr=False)
    censored: np.ndarray = field(repr=False)   # dp_norm at or below 1, value unknown
    abs_day: np.ndarray = field(repr=False)

    @property
    def id(self) -> str:
        return f"{self.train}:{self.start}-{self.end}"

    @property
    def length(self) -> int:
        return int(self.end - self.start)

    @property
    def n(self) -> int:
        return int(len(self.day))


def split_cycles(daily: pd.DataFrame, cfg: RODepositConfig | None = None) -> list[Cycle]:
    """Cut each train's record at its cleaning campaigns.

    Days inside a campaign (between its first and last leg) belong to no cycle.
    """
    cfg = cfg or RODepositConfig()
    legs = cleaning_legs(daily)
    cycles: list[Cycle] = []
    for t, g in daily.groupby("train"):
        t = str(t)
        camps = group_campaigns(legs.get(t, []), cfg.campaign_gap_days)
        first, last = int(g["day_index"].min()), int(g["day_index"].max())
        bounds = []
        if camps:
            bounds.append((first, camps[0][0], "initial"))
            for (a0, a1), (b0, _) in zip(camps[:-1], camps[1:]):
                bounds.append((a1, b0, "complete"))
            bounds.append((camps[-1][1], last + 1, "open"))
        else:
            bounds.append((first, last + 1, "initial"))
        for s, e, kind in bounds:
            lo = s if kind == "initial" else s + 1
            seg = g[(g["day_index"] >= lo) & (g["day_index"] < e)]
            cycles.append(Cycle(
                train=t, start=int(s), end=int(e), kind=kind,
                day=(seg["day_index"].to_numpy() - s).astype(float),
                dp_norm=seg["dp_norm"].to_numpy(float),
                censored=seg["censored"].to_numpy(bool),
                abs_day=seg["day_index"].to_numpy(int)))
    return cycles


# ---------------------------------------------------------------- the audit

def recovery_by_campaign(daily: pd.DataFrame, ledger: pd.DataFrame, window: int = 14,
                         min_points: int = 3, column: str = "dp_pct") -> pd.DataFrame:
    """Median before [d0-window, d0-1] and after [d0+1, d0+window] each campaign.

    This is the published event-pair construction: a campaign needs at least
    `min_points` days on each side to be evaluable.
    """
    rows = []
    for _, r in ledger.iterrows():
        g = daily[daily["train"] == r["train"]].set_index("day_index")[column]
        d0 = int(r["day_index"])
        pre = g[(g.index >= d0 - window) & (g.index <= d0 - 1)]
        post = g[(g.index >= d0 + 1) & (g.index <= d0 + window)]
        ok = len(pre) >= min_points and len(post) >= min_points
        rows.append(dict(train=r["train"], episode=r["episode"], day=d0, stage=r["stage"],
                         n_pre=len(pre), n_post=len(post), evaluable=ok,
                         pre=float(pre.median()) if len(pre) else np.nan,
                         post=float(post.median()) if len(post) else np.nan))
    out = pd.DataFrame(rows)
    out["drop"] = out["pre"] - out["post"]
    return out


def duty_cycle(daily: pd.DataFrame, level: float, window: int = 7) -> float:
    """Share of observed days on which the trailing `window`-day median of
    DP_norm is at or above `level` - the published 'duty' of a threshold rule."""
    hit = tot = 0
    for _, g in daily.groupby("train"):
        s = g.set_index("day_index")["dp_norm"]
        full = s.reindex(range(int(s.index.min()), int(s.index.max()) + 1))
        med = full.rolling(window, min_periods=1).median()[full.notna()]
        hit += int((med >= level).sum())
        tot += int(med.notna().sum())
    return hit / tot if tot else float("nan")


def audit(daily: pd.DataFrame, ledger: pd.DataFrame,
          cfg: RODepositConfig | None = None) -> dict:
    """Everything checked before any model. Returns plain numbers for the record."""
    cfg = cfg or RODepositConfig()
    legs = cleaning_legs(daily)
    camps = {t: group_campaigns(v, cfg.campaign_gap_days) for t, v in legs.items()}
    trains = {}
    for t, g in daily.groupby("train"):
        d = g["day_index"].to_numpy()
        gaps = np.diff(d)
        trains[str(t)] = dict(
            rows=int(len(g)), first=int(d.min()), last=int(d.max()),
            coverage=round(len(g) / (d.max() - d.min() + 1), 3),
            gaps_over_7=int((gaps > 7).sum()), longest_gap=int(gaps.max()),
            censored_share=round(float(g["censored"].mean()), 3),
            legs=len(legs.get(str(t), [])), campaigns=len(camps.get(str(t), [])))
    rec = recovery_by_campaign(daily, ledger)
    ev = rec[rec["evaluable"]]
    # ndp_pct right after each leg: a per-cleaning reset floors it at zero
    post_ndp, post_dp = [], []
    for t, L in legs.items():
        g = daily[daily["train"] == t].set_index("day_index")
        for c in L:
            w = g[(g.index > c) & (g.index <= c + 14)]
            post_ndp += list(w["ndp_pct"])
            post_dp += list(w["dp_pct"])
    ledger_days = {t: sorted(ledger.loc[ledger["train"] == t, "day_index"].astype(int))
                   for t in ledger["train"].unique()}
    camps_match = all([a for a, _ in camps.get(t, [])] == ledger_days.get(t, [])
                      for t in set(camps) | set(ledger_days))
    return dict(
        rows=int(len(daily)),
        trains=trains,
        legs_total=int(sum(len(v) for v in legs.values())),
        campaigns_total=int(sum(len(v) for v in camps.values())),
        ledger_campaigns=int(len(ledger)),
        campaigns_match_ledger=bool(camps_match),
        over_max_days=int(daily["over_max"].sum()),
        over_max_max=int(daily["n_10min_samples"].max()),
        censored_share=round(float(daily["censored"].mean()), 3),
        indicator_zero_share={c: round(float((daily[c] <= 0).mean()), 3)
                              for c in ("ndp_pct", "dp_pct", "flux_pct", "sp_pct")},
        post_cleaning_median=dict(ndp_pct=round(float(np.median(post_ndp)), 4),
                                  dp_pct=round(float(np.median(post_dp)), 4)),
        recovery_evaluable=int(len(ev)),
        recovery_down=int((ev["drop"] > 0).sum()),
        duty={lvl: round(duty_cycle(daily, lvl), 3) for lvl in (1.10, 1.15, 1.20)},
    )
