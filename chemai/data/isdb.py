"""ISDB - International Stiction Data Base (Jelali & Huang, 2010).

File: isdb10.mat from https://sites.ualberta.ca/~bhuang/ISDB.zip. Not committed;
place it in data/isdb/. The manual asks users to cite the book, its homepage
or the original source.

Labels are read from the PUBLISHED comments, nothing inferred
(projects/p02_control_loops/VALIDATION.md, section 12):

    tier "stated"  the comment states the diagnosis
    tier "likely"  the comment says "(likely)" - all 17 from one contributor;
                   kept, but always reported separately from "stated"
    label None     no diagnosis in the comment (53 loops)

"No stiction" is NOT mapped to healthy: one such loop is SACAC's tight-tuning file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
import re

import numpy as np
import pandas as pd

from chemai.config import DATA_DIR

log = logging.getLogger(__name__)

_TYPES = (("flow", "F"), ("level", "L"), ("pressure", "P"), ("temperature", "T"),
          ("concentration", "Q"), ("analyser", "Q"), ("analyzer", "Q"),
          ("composition", "Q"), ("thickness", "X"))

# SACAC files that are ISDB loops, value-identical (VALIDATION.md 12.3).
ISDB_IN_SACAC: dict[str, str] = {
    "chemicals.10": "stiction-P-chemical-baccidicapaci-2018",
    "chemicals.25": "stiction-P-oilgas-baccidicapaci-2018",
    "power.4": "stiction-L-power-baccidicapaci-2018",
    "pulpPapers.3": "stiction-L-paper-horch-2003",
    "pulpPapers.6": "tuning-L-paper-horch-2003",
    "chemicals.14": "sensor-F-oilgas-thornhill-2007",
    "chemicals.15": "unknown-P-oilgas-thornhill-2007-1",
    "chemicals.16": "unknown-P-oilgas-thornhill-2007-2",
    "chemicals.54": "unknown-L-oilgas-thornhill-2002",
    "chemicals.41": "saturation-T-oilgas-thornhill-2002",
}


@dataclass
class IsdbLoop:
    key: str                # "chemicals.12"
    sector: str
    number: int
    loop_type: str          # F, L, P, T, Q, X or "?"
    contributor: str
    label: str | None       # stiction, no_stiction, no_oscillation, tuning, external_oscillation
    tier: str | None        # stated, likely, mixed
    ts: float
    comment: str
    data: pd.DataFrame = field(repr=False)

    @property
    def in_sacac(self) -> str | None:
        return ISDB_IN_SACAC.get(self.key)


def label_from_comment(text: str) -> tuple[str | None, str | None]:
    """(label, tier) from the published comment. Order matters: 'tuning issue,
    no stiction' is a tuning label, not a no-stiction one."""
    t = text.lower()
    tuning = re.search(r"\btuning\b", t) is not None   # not 'detuning' (buildings.7)
    if "stiction (likely)" in t:
        return "stiction", "likely"
    if "disturbance (likely)" in t:
        return "external_oscillation", "likely"
    if tuning and "stiction" in t.replace("no stiction", ""):
        return "stiction", "mixed"
    if tuning or "marginal stability" in t or "dead zone" in t:
        return "tuning", "stated"
    if "no stiction" in t:
        return "no_stiction", "stated"
    if "no oscillation" in t:
        return "no_oscillation", "stated"
    if "stiction" in t:
        return "stiction", "stated"
    if "external disturbance" in t:
        return "external_oscillation", "stated"
    return None, None


def _loop_type(brief: str) -> str:
    b = brief.lower()
    return next((code for word, code in _TYPES if word in b), "?")


def _text(x) -> str:
    return " ".join(str(s).strip() for s in np.atleast_1d(x))


def load_isdb(path: Path | str | None = None, include_test: bool = False) -> list[IsdbLoop]:
    """All ISDB loops. The six synthetic 'testdata' records are excluded unless asked."""
    import scipy.io as sio

    path = Path(path) if path else DATA_DIR / "isdb" / "isdb10.mat"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - ISDB is not committed; see README")
    cdata = sio.loadmat(path, squeeze_me=True, struct_as_record=False)["cdata"]
    loops = []
    for sector in cdata._fieldnames:
        if sector == "testdata" and not include_test:
            continue
        sec = getattr(cdata, sector)
        for name in sec._fieldnames:
            l = getattr(sec, name)
            brief = str(l.BriefComments).strip()
            comment = f"{brief} | {_text(l.Comments)}"
            pv = np.atleast_1d(np.asarray(l.PV, dtype=float))
            n = len(pv)

            def col(name):
                v = np.atleast_1d(np.asarray(getattr(l, name), dtype=float)) \
                    if name in l._fieldnames else np.array([])
                return v if len(v) == n else np.full(n, np.nan)

            label, tier = label_from_comment(comment)
            m = re.findall(r"\(([^()]*[A-Z]\.[^()]*)\)", brief)
            number = int(re.sub(r"\D", "", name) or 0)
            loops.append(IsdbLoop(
                key=f"{sector}.{number}", sector=sector, number=number,
                loop_type=_loop_type(brief), contributor=m[-1].strip() if m else "?",
                label=label, tier=tier, ts=float(l.Ts), comment=comment,
                data=pd.DataFrame({"sp": col("SP"), "pv": pv, "op": col("OP")})))
    log.info("loaded %d ISDB loops (%d labelled stated, %d likely)", len(loops),
             sum(x.tier == "stated" for x in loops), sum(x.tier == "likely" for x in loops))
    return loops
