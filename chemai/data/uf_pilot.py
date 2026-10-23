"""The public UF pilot record - file discovery, readers, the clock audit, episodes.

Source: Zenodo record 4630769 (CC0), also Dryad doi:10.5068/D1310B. An
ultrafiltration (UF) unit pre-treating seawater for a reverse-osmosis pilot,
180 run files, 2012-2015, split by the depositors into `training/` (150) and
`test/` (30). Not committed: `extract_archive` unpacks the rar the record
publishes, after checking the md5 printed on the record page.

What the files are, as found by reading all 180 (projects/p04_tide_fouling/VALIDATION.md 1):

  * Four layouts. 146 csv files carry a logged clock ("Time", an Excel day
    number) and 16 sensor tags; 1 csv carries the same columns without a
    header; 30 files (xlsx, or csv saved from a sheet) carry no clock at all;
    3 are xlsx files with a .csv name. Only files with a clock can be placed
    against the tide.
  * Six clocked files start on a different day than their name says - one a
    year off, one with day and month swapped. They are excluded rather than
    guessed at.
  * The turbidity tag reads -2.5 (no signal) in most runs, and in some runs it
    reads filtrate (about 0.03 NTU), not feed. Only runs whose median is at
    least 0.2 NTU are treated as feed turbidity.
  * Three files of the depositors' `test/` folder continue a `training/` run
    without a break: the published split is not separated in time.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import logging
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import pandas as pd

from chemai.config import DATA_DIR, UFPilotConfig

log = logging.getLogger(__name__)

EXCEL_EPOCH = pd.Timestamp("1899-12-30")
SPLITS = ("training", "test")

# Instrument tags as logged, and the names used everywhere else. The meanings
# come from the record's own header rows (the sheets that spell them out).
TAGS = {
    "FE/FT-100": "q_uf",          # UF inflow, gpm
    "FE/FT-101": "q_e1",          # element 1 inflow, gpm
    "FE/FT-102": "q_e2",
    "FE/FT-103": "q_e3",
    "FE/FT-104": "q_bw",          # backwash flow, gpm
    "pH/pHT-100": "ph",
    "ORP/ORPT-100": "orp",
    "PT-100": "p_mf_dp",          # MF trans-filter pressure, psig
    "PT-101": "p_uf_in",          # UF inlet (filtration) pressure, psig
    "PT-102": "p_bw_feed",        # UF feed-side backwash pressure
    "PT-103": "p_bw_filt",        # UF filtrate-side backwash pressure
    "PT-104": "p_uf_out",         # UF filtrate-side pressure, psig
    "PT-100x": "p_mf_in",         # MF inlet pressure (feed pump discharge), psig
    "Tu/TuT-100": "turbidity",    # NTU; -2.5 means no signal
    "VFD-100-FB": "pump_speed",   # feed pump drive feedback
    "TE/TT-100": "temp_c",        # filtrate temperature, deg C
}
TAG_ORDER = list(TAGS)


# ------------------------------------------------------------------ files

def data_dir(cfg: UFPilotConfig | None = None) -> Path:
    cfg = cfg or UFPilotConfig()
    return DATA_DIR / cfg.folder


def md5_of(path: Path | str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_archive(dest: Path | str | None = None, cfg: UFPilotConfig | None = None,
                     timeout: float = 600.0) -> Path:
    """Fetch the 356 MB rar from Zenodo once (it is not committed)."""
    import urllib.request
    cfg = cfg or UFPilotConfig()
    dest = Path(dest) if dest else data_dir(cfg)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / cfg.archive
    if not path.exists():
        log.info("downloading %s", cfg.url)
        with urllib.request.urlopen(cfg.url, timeout=timeout) as r, open(path, "wb") as f:   # noqa: S310
            shutil.copyfileobj(r, f, length=1 << 20)
    return path


def extract_archive(archive: Path | str, dest: Path | str | None = None,
                    cfg: UFPilotConfig | None = None) -> Path:
    """Check the archive's md5 and unpack it (bsdtar, unrar or 7z - whichever exists).

    A file that does not match the published md5 is refused: an analysis built
    on a silently different file cannot be reproduced by anyone else.
    """
    cfg = cfg or UFPilotConfig()
    archive = Path(archive)
    got = md5_of(archive)
    if got != cfg.archive_md5:
        raise ValueError(f"md5 mismatch for {archive.name}: {got} != {cfg.archive_md5}")
    dest = Path(dest) if dest else data_dir(cfg)
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / cfg.inner / "training").is_dir():
        return dest / cfg.inner
    for tool, args in (("bsdtar", ["-xf", str(archive), "-C", str(dest)]),
                       ("unrar", ["x", "-o+", "-inul", str(archive), str(dest) + "/"]),
                       ("7z", ["x", "-y", f"-o{dest}", str(archive)])):
        exe = shutil.which(tool)
        if exe:
            subprocess.run([exe, *args], check=True)
            return dest / cfg.inner
    raise RuntimeError("no RAR reader found - install libarchive-tools (bsdtar), unrar or p7zip")


def run_root(root: Path | str | None = None, cfg: UFPilotConfig | None = None) -> Path:
    cfg = cfg or UFPilotConfig()
    root = Path(root) if root else data_dir(cfg)
    if (root / cfg.inner).exists():
        root = root / cfg.inner
    if not all((root / s).is_dir() for s in SPLITS):
        raise FileNotFoundError(
            f"{root} does not hold training/ and test/. The record is not committed - "
            f"download {cfg.url} and call extract_archive().")
    return root


def file_kind(path: Path) -> str:
    """Which of the record's layouts a file uses (the extension is not reliable)."""
    with open(path, "rb") as f:
        head = f.read(200)
    if head[:2] == b"PK":
        return "sheet"                                   # xlsx, whatever the name says
    text = head.decode("latin-1")
    if text.startswith("Time,Time Elapsed"):
        return "csv_clock"
    if re.match(r"^\d{5}\.\d+,", text):
        return "csv_clock_noheader"
    return "csv_noclock"


def name_date(path: Path) -> datetime:
    return datetime.strptime(path.stem[:8], "%Y%m%d")


def list_files(root: Path | str | None = None, cfg: UFPilotConfig | None = None) -> pd.DataFrame:
    root = run_root(root, cfg)
    rows = []
    for split in SPLITS:
        for p in sorted((root / split).iterdir()):
            rows.append(dict(split=split, file=p.name, path=str(p), kind=file_kind(p),
                             name_date=name_date(p), bytes=p.stat().st_size))
    return pd.DataFrame(rows)


def read_run(path: Path | str) -> pd.DataFrame:
    """Read one clocked run file into standard column names, untouched otherwise.

    Index: logged local time. Raises on files without a clock. Malformed lines
    (three 2012 files hold rows with 39 fields where 18 belong) are skipped and
    counted by `count_bad_lines`, never repaired.
    """
    path = Path(path)
    kind = file_kind(path)
    if kind == "csv_clock":
        df = pd.read_csv(path, skiprows=[1], low_memory=False, on_bad_lines="skip")
    elif kind == "csv_clock_noheader":
        df = pd.read_csv(path, header=None, low_memory=False, on_bad_lines="skip")
        df.columns = ["Time", "Time Elapsed"] + TAG_ORDER[: df.shape[1] - 2]
    else:
        raise ValueError(f"{path.name} has no logged clock ({kind})")
    t = pd.to_numeric(df.pop("Time"), errors="coerce")
    df = df.drop(columns=["Time Elapsed"]).rename(columns=TAGS)
    df = df.apply(pd.to_numeric, errors="coerce")
    df.index = EXCEL_EPOCH + pd.to_timedelta(t.to_numpy(float), unit="D")
    df.index.name = "t_local"
    return df[~df.index.isna()]


def count_bad_lines(path: Path | str) -> int:
    """Lines whose field count differs from the header's (they are skipped on reading)."""
    path = Path(path)
    if not file_kind(path).startswith("csv_clock"):
        return 0
    with open(path, "rb") as f:
        lines = f.read().split(b"\n")
    lines = [ln for ln in lines if ln.strip()]
    n = lines[0].count(b",")
    return sum(1 for ln in lines[1:] if ln.count(b",") != n)


# ------------------------------------------------------------- the clock

def second_sunday_march(year: int) -> datetime:
    d = datetime(year, 3, 1)
    first_sunday = d + timedelta(days=(6 - d.weekday()) % 7)
    return first_sunday + timedelta(days=7)


def first_sunday_november(year: int) -> datetime:
    d = datetime(year, 11, 1)
    return d + timedelta(days=(6 - d.weekday()) % 7)


def in_daylight_saving(t: pd.DatetimeIndex) -> np.ndarray:
    """True where the local clock is one hour ahead of standard time.

    The convention: from 02:00 on the second Sunday of March to 02:00 on the
    first Sunday of November. The hour that repeats in November is read as the
    standard-time one; nothing in this record falls inside it.
    """
    t = pd.DatetimeIndex(t)
    out = np.zeros(len(t), bool)
    for y in np.unique(t.year):
        on = second_sunday_march(int(y)) + timedelta(hours=2)
        off = first_sunday_november(int(y)) + timedelta(hours=1)    # 02:00 daylight = 01:00 standard
        sel = t.year == y
        out[sel] = (t[sel] >= on) & (t[sel] < off)
    return out


def to_standard_time(t_local, cfg: UFPilotConfig | None = None) -> pd.DatetimeIndex:
    """Local civil clock -> local standard clock (one clock for every run)."""
    cfg = cfg or UFPilotConfig()
    t = pd.DatetimeIndex(t_local)
    if not cfg.remove_daylight_saving:
        return t
    return t - pd.to_timedelta(in_daylight_saving(t).astype(int), unit="h")


# ------------------------------------------------------------- the audit

@dataclass
class RunInfo:
    split: str
    file: str
    path: str
    kind: str
    name_date: datetime
    t_start: pd.Timestamp | None = None
    t_end: pd.Timestamp | None = None
    n: int = 0
    dt_median_s: float = float("nan")

    @property
    def hours(self) -> float:
        if self.t_start is None:
            return 0.0
        return (self.t_end - self.t_start).total_seconds() / 3600

    @property
    def clock_offset_days(self) -> float:
        if self.t_start is None:
            return float("nan")
        return (self.t_start - pd.Timestamp(self.name_date)).total_seconds() / 86400


def audit_files(root: Path | str | None = None, cfg: UFPilotConfig | None = None) -> pd.DataFrame:
    """One row per file: layout, logged span, sampling, and whether its clock is trusted."""
    cfg = cfg or UFPilotConfig()
    files = list_files(root, cfg)
    rows = []
    for _, f in files.iterrows():
        info = RunInfo(f.split, f.file, f.path, f.kind, f.name_date)
        if f.kind.startswith("csv_clock"):
            d = read_run(f.path)
            if len(d):
                info.t_start, info.t_end, info.n = d.index[0], d.index[-1], len(d)
                info.dt_median_s = float(np.median(np.diff(d.index.as_unit("ns").asi8)) / 1e9) if len(d) > 1 else np.nan
        rows.append(dict(split=info.split, file=info.file, path=info.path, kind=info.kind,
                         name_date=info.name_date, t_start=info.t_start, t_end=info.t_end,
                         n=info.n, hours=info.hours, dt_median_s=info.dt_median_s,
                         clock_offset_days=info.clock_offset_days,
                         bad_lines=count_bad_lines(f.path)))
    out = pd.DataFrame(rows)
    lo, hi = cfg.clock_window_days
    out["has_clock"] = out.t_start.notna()
    out["clock_trusted"] = out.has_clock & (out.clock_offset_days >= lo) & (out.clock_offset_days < hi)
    out["long_enough"] = out.hours >= cfg.min_run_hours
    out["included"] = out.clock_trusted & out.long_enough
    return out


def assign_episodes(runs: pd.DataFrame, cfg: UFPilotConfig | None = None) -> pd.Series:
    """Chain runs whose logged start follows the previous end within the gap.

    An episode is one stretch of continuous operation. It is the unit the
    tide test shifts (VALIDATION.md 3): runs inside an episode share whatever
    slow weather or operation the plant was going through.
    """
    cfg = cfg or UFPilotConfig()
    r = runs.sort_values("t_start")
    gap = pd.Timedelta(minutes=cfg.episode_gap_minutes)
    ep, last_end, ids = 0, None, {}
    for i, row in r.iterrows():
        if last_end is None or row.t_start - last_end > gap:
            ep += 1
        ids[i] = ep
        last_end = row.t_end if last_end is None or row.t_end > last_end else last_end
    return pd.Series(ids).reindex(runs.index)


def episode_starts(runs: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    """Per episode: when it started and how long the unit had stood idle before.

    `runs` are the kept runs (with an `episode` column); `audit` is the full
    file audit, so a short run that was dropped from the analysis still counts
    as the unit having run (it shortens the idle time, it does not hide it).
    """
    clocked = audit[audit.clock_trusted].sort_values("t_start")
    rows = []
    for e, g in runs.groupby("episode"):
        start = g.t_start.min()
        before = clocked[clocked.t_start < start]
        idle = (start - before.t_end.max()).total_seconds() / 3600 if len(before) else np.inf
        rows.append({"episode": e, "episode_start": start, "idle_before_h": max(idle, 0.0)})
    return pd.DataFrame(rows)
