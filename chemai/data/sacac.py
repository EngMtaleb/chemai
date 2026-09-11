"""SACAC PID data repository - loader with the data-quality rules built in.

Source: sacac.org.za/resources/ ; Bauer et al. (2019), Ind. Eng. Chem. Res.
58, 11430-11439. Data is not committed - place the extracted repository in
data/sacac/.

Every rule here was found by reading the files, and each is documented in
projects/p02_control_loops/VALIDATION.md:

  * columns are read by NAME, case-insensitive: some files put PV before SP,
    one has PV only, several add an error column
  * bare '\\r' line endings are normalised before parsing
  * the time column is not a clock (restyled or irregular) - kept as
    `time_raw`, never used for arithmetic
  * rows listed in SACAC_KEEP_ROWS are cut here; the raw file is never edited
  * seven 'DB-N' runs of one loop share one loop_id
"""
from __future__ import annotations

from dataclasses import dataclass, field
from io import StringIO
import logging
from pathlib import Path
import re

import numpy as np
import pandas as pd

from chemai.config import DATA_DIR, SACAC_DEAD_TIME, SACAC_KEEP_ROWS, SACAC_LABELS

log = logging.getLogger(__name__)

_SIGNALS = ("sp", "pv", "op")


@dataclass
class LoopRecord:
    name: str               # file stem
    category: str           # folder category from the filename: stiction, tuning, ...
    loop_type: str          # F, L, P, Q, T
    source: str             # horch, thornhill, baccidicapaci, bauer, ...
    loop_id: str            # group for any split: DB runs share one
    label: str | None       # study label from SACAC_LABELS, None if not assigned
    data: pd.DataFrame = field(repr=False)
    dead_time: tuple[float, float] | None = None
    rows_dropped: int = 0
    ts_meta: float | None = None     # sampling interval stated in the .txt, s

    @property
    def ts_column(self) -> float | None:
        """Median step of the time column, where it is numeric. NOT necessarily
        seconds - Thornhill files count samples - so never use it alone."""
        t = pd.to_numeric(self.data["time_raw"], errors="coerce")
        if t.isna().any() or len(t) < 2:
            return None
        return float(np.median(np.diff(t.to_numpy())))

    @property
    def ts(self) -> float | None:
        """Sampling interval, s - only when the description and the time
        column agree. Disagreement returns None: guessing a sampling time
        silently corrupts any delay-based index. Irregular steps also
        return None (tuning-L-paper-horch steps 1-4 s)."""
        t = pd.to_numeric(self.data["time_raw"], errors="coerce")
        regular = t.notna().all() and np.ptp(np.diff(t.to_numpy())) == 0
        if self.ts_meta is not None and regular and self.ts_column == self.ts_meta:
            return self.ts_meta
        return None


def read_loop_csv(path: Path | str) -> pd.DataFrame:
    """Read one single-loop file into columns time_raw, sp, pv, op (+ error).

    Missing signals become NaN columns rather than errors, so the caller
    decides what a file without OP can be used for.
    """
    raw = Path(path).read_bytes().decode("utf-8-sig")
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    header = text.split("\n", 1)[0]
    sep = ";" if header.count(";") >= header.count(",") else ","
    df = pd.read_csv(StringIO(text), sep=sep)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "time" not in df.columns:
        raise ValueError(f"{Path(path).name}: no time column in {list(df.columns)}")

    out = pd.DataFrame({"time_raw": df["time"]})
    for s in _SIGNALS:
        out[s] = pd.to_numeric(df[s], errors="coerce") if s in df.columns else np.nan
    if "error" in df.columns:
        out["error"] = pd.to_numeric(df["error"], errors="coerce")
    return out


_NAME = re.compile(r"^(?P<cat>[a-z]+)-(?P<lt>[A-Z])-(?P<rest>.+)$", re.IGNORECASE)
_SOURCES = ("baccidicapaci", "thornhill", "horch", "bauer", "brooks", "lindner")


def parse_name(stem: str) -> dict:
    """Category, loop type, source and loop_id from a SACAC file stem."""
    m = _NAME.match(stem)
    if not m:
        raise ValueError(f"unrecognised SACAC file name '{stem}'")
    cat = m["cat"].lower().replace("quantization", "quantisation")
    source = next((s for s in _SOURCES if s in stem.lower()), "unknown")
    loop_id = re.sub(r"-DB-\d+-", "-DB-", stem)          # runs of one loop
    return {"category": cat, "loop_type": m["lt"].upper(), "source": source,
            "loop_id": loop_id}


def _label_for(stem: str, category: str) -> str | None:
    if stem in SACAC_LABELS:
        return SACAC_LABELS[stem]
    for key, lab in SACAC_LABELS.items():            # prefix keys (DB families)
        if stem.startswith(key):
            return lab
    return SACAC_LABELS.get(category)


_RATE = re.compile(r"sampling rate:\s*([\d.]+)\s*(s|sec|second|min|minute)", re.IGNORECASE)


def read_sampling_rate(folder: Path) -> float | None:
    """Sampling interval in seconds from the description file in `folder`,
    or None when it is absent or stated as non-constant."""
    for txt in sorted(folder.glob("*.txt")):
        m = _RATE.search(txt.read_bytes().decode("utf-8", "ignore"))
        if m:
            return float(m[1]) * (60.0 if m[2].lower().startswith("min") else 1.0)
    return None


def load_record(path: Path | str) -> LoopRecord:
    path = Path(path)
    stem = path.stem
    info = parse_name(stem)
    df = read_loop_csv(path)
    dropped = 0
    if stem in SACAC_KEEP_ROWS:
        a, b = SACAC_KEEP_ROWS[stem]
        dropped = len(df) - (min(b, len(df)) - a)
        df = df.iloc[a:b].reset_index(drop=True)
        log.warning("%s: kept rows %d-%d, dropped %d (data-quality rule)", stem, a, b - 1, dropped)
    rec = LoopRecord(name=stem, data=df, label=_label_for(stem, info["category"]),
                     dead_time=SACAC_DEAD_TIME.get(stem), rows_dropped=dropped,
                     ts_meta=read_sampling_rate(path.parent), **info)
    if rec.ts_meta is not None and rec.ts_column not in (None, rec.ts_meta):
        log.info("%s: description says %.0f s, time column steps %s - ts left undefined",
                 stem, rec.ts_meta, rec.ts_column)
    return rec


def load_sacac(root: Path | str | None = None) -> list[LoopRecord]:
    """All single-loop files under `root` (plant-wide folders are skipped:
    different format, different question)."""
    root = Path(root) if root else DATA_DIR / "sacac"
    if not root.exists():
        raise FileNotFoundError(f"{root} not found - SACAC data is not committed; see README")
    files = sorted(p for p in root.rglob("*.csv")
                   if "plantwide" not in p.relative_to(root).as_posix().lower())
    recs = [load_record(p) for p in files]
    log.info("loaded %d SACAC loop files, %d distinct loops",
             len(recs), len({r.loop_id for r in recs}))
    return recs
