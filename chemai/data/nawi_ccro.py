"""The public closed-circuit RO (CCRO) testbed record - checking, converting, reading.

Source: doi:10.7481/1963162 (CC BY 4.0), "Foundational Control Methods for
Water Treatment Systems - Online Actuator and Sensor Data". Seven zip files,
351 MB, about 3.4 GB of CSV once unpacked. Not committed: download the zips
from the record into data/nawi_ccro/ and `check_files` verifies them.

What the files are, as found by reading all of them (VALIDATION.md 1):

  * A brackish NaCl feed (about 1.7-2.8 mS/cm) in the experiments used here,
    a closed loop of about 15-20 L, one spiral-wound element, a feed pump that
    tops the loop up and a recirculation pump. One sample every ~0.5 s.
  * One CSV per batch. Each starts with the drain of the previous batch's
    brine ("Discharge"), then start-up, stabilisation and filtration
    ("Experiment") until the reject conductivity reaches the batch's
    critical value. So a batch's brine is measured at the start of the
    next file.
  * Pump power comes from the drives and is approximate; at this scale the
    pumps run far below their best efficiency, so the specific energy is
    ~10 kWh/m3, ten times a full-scale plant. The shape of the curve, not
    its level, is what transfers.
  * The feed-conductivity column reads a sensor whose range starts at
    15 mS/cm; at a 2 mS/cm feed it is not used.

The zips are read as text only. Nothing inside them is executed or unpickled
(one experiment folder ships a pickle; it is skipped).
"""
from __future__ import annotations

import hashlib
import io
import logging
from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd

from chemai.config import CCROConfig, DATA_DIR

log = logging.getLogger(__name__)

# Text columns; everything else is numeric.
STATE_COLUMNS = {"STATE", "STATUS", "CCD STATE"}
TEXT_COLUMNS = STATE_COLUMNS | {
    "Brine Discharge Valve Position", "Feed Flush Valve Position", "Permeate Transfer Valve State",
    "Acid Pump State", "Anti-Scalant Pump State", "Salt Pump State", "SYSTEM POWER STATE",
}
# The two oldest datasets spell some headers differently.
RENAME = {
    "Reject Conductivity (mS.cm)": "Reject Conductivity (mS/cm)",
    "Water Flux (LMH)": "Flux (LMH)",
    "Approx. Feed Pump Power Consumption (kW)": "Feed Pump Power Draw (kW)",
    "Approx. Recirc. Pump Power Consumption (kW)": "Recirculation Pump Power Draw (kW)",
    "STATUS": "STATE",
    "CCD STATE": "STATE",
}
CATEGORIES = ("dataset", "exp", "file", "state")


def data_dir(cfg: CCROConfig | None = None) -> Path:
    cfg = cfg or CCROConfig()
    return DATA_DIR / cfg.folder


def parquet_dir(cfg: CCROConfig | None = None) -> Path:
    return data_dir(cfg) / "parquet"


def sha256_of(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_files(cfg: CCROConfig | None = None) -> pd.DataFrame:
    """Every expected zip: present, right size, right sha256."""
    cfg = cfg or CCROConfig()
    rows = []
    for name, size, sha in cfg.files:
        p = data_dir(cfg) / name
        present = p.exists()
        rows.append({
            "file": name, "url": cfg.base_url + name.replace(" ", "%20"),
            "present": present,
            "size_ok": present and p.stat().st_size == size,
            "sha256_ok": present and sha256_of(p) == sha,
        })
    return pd.DataFrame(rows)


def snake(column: str) -> str:
    """'Feed Pump Power Draw (kW)' -> 'feed_pump_power_draw_kw'."""
    c = column.replace("%", "pct").replace("/", "_per_").replace("^3", "3")
    return re.sub(r"[^0-9a-zA-Z]+", "_", c).strip("_").lower()


def read_csv_bytes(raw: bytes) -> pd.DataFrame | None:
    """One batch file as logged -> typed frame with a timestamp; None if empty."""
    if len(raw) < 50:
        return None
    text = raw.decode("utf-8", "replace")
    skip = 1 if text.startswith("Date:,") else 0          # the oldest files carry a title line
    df = pd.read_csv(io.StringIO(text), skiprows=skip, low_memory=False,
                     na_values=["NaN", "Inf", "-Inf", "inf", "-inf"])
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns=RENAME)
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.dropna(how="all")
    if df.empty or "Date" not in df or "Time" not in df:
        return None
    out = pd.DataFrame({"ts": pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str),
                                             format="mixed", errors="coerce")})
    for c in df.columns:
        if c in ("Date", "Time"):
            continue
        if c in TEXT_COLUMNS or c == "STATE":
            out[snake(c)] = df[c].astype(object).where(df[c].notna(), None)
        else:
            out[snake(c)] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    return out


def batch_id(member: str) -> tuple[str, str, str, int]:
    """'Dataset6_.../Exp2/Dataset14.csv' -> ('dataset6', 'Exp2', 'Dataset14.csv', 14)."""
    parts = member.split("/")
    m = re.match(r"Dataset(\d+)", parts[0])
    ds = f"dataset{m.group(1)}" if m else parts[0].lower()
    exp = "/".join(parts[1:-1])
    m2 = re.match(r"Dataset(\d+)\.csv$", parts[-1])
    return ds, exp, parts[-1], int(m2.group(1)) if m2 else -1


def convert_zip(path: Path | str) -> pd.DataFrame:
    """All CSV members of one dataset zip -> one long frame, rows in logged order."""
    z = zipfile.ZipFile(path)
    frames = []
    for member in z.namelist():
        if not member.lower().endswith(".csv"):
            continue
        df = read_csv_bytes(z.read(member))
        if df is None:
            log.info("empty: %s", member)
            continue
        ds, exp, fname, seq = batch_id(member)
        df.insert(0, "row", np.arange(len(df), dtype=np.int32))
        df.insert(0, "seq", np.int32(seq))
        df.insert(0, "file", fname)
        df.insert(0, "exp", exp)
        df.insert(0, "dataset", ds)
        frames.append(df)
    big = pd.concat(frames, ignore_index=True, sort=False)
    for c in big.columns:
        if c in CATEGORIES or (big[c].dtype == object and c not in ("ts",)):
            big[c] = big[c].astype("category")
        elif big[c].dtype == "float64":
            big[c] = big[c].astype("float32")
    return big


def convert_all(cfg: CCROConfig | None = None, overwrite: bool = False) -> list[Path]:
    """Each dataset zip -> data/nawi_ccro/parquet/datasetN.parquet."""
    cfg = cfg or CCROConfig()
    out_dir = parquet_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, _, _ in cfg.files:
        if not name.startswith("Dataset"):
            continue
        n = re.match(r"Dataset(\d+)", name).group(1)
        target = out_dir / f"dataset{n}.parquet"
        if target.exists() and not overwrite:
            written.append(target)
            continue
        big = convert_zip(data_dir(cfg) / name)
        big.to_parquet(target, compression="zstd", index=False)
        log.info("%s: %d rows -> %s", name, len(big), target)
        written.append(target)
    return written


def read_dataset(name: str, columns: list[str] | None = None,
                 cfg: CCROConfig | None = None) -> pd.DataFrame:
    """One converted dataset ('dataset3' ...), optionally only some columns."""
    p = parquet_dir(cfg) / f"{name}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"{p} - run convert_all() after check_files()")
    return pd.read_parquet(p, columns=columns)
