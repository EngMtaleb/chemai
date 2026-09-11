"""ISDB reader - labels come from the published comment, and nothing else."""
import numpy as np
import pytest
import scipy.io as sio

from chemai.data import label_from_comment, load_isdb


@pytest.mark.parametrize("comment,expected", [
    ("Flow control (FC145); with stiction (A. Horch)", ("stiction", "stated")),
    ("Level control; with tuning problem. | There is a tuning issue, no stiction.", ("tuning", "stated")),
    ("Level control (lev3horch); no stiction. (A. Horch)", ("no_stiction", "stated")),
    ("Temperature control; no oscillation (A. Singhal)", ("no_oscillation", "stated")),
    ("Flow control (FC1) with stiction (likely) (C. Scali)", ("stiction", "likely")),
    ("Flow control with disturbance (likely) (C. Scali)", ("external_oscillation", "likely")),
    ("Temperature control; with stiction and tight tuning (A. Singhal)", ("stiction", "mixed")),
    # 'detuning' is not a tuning diagnosis (buildings.7)
    ("Temperature control; with stiction | ... after detuning the controller.", ("stiction", "stated")),
    ("Pressure control (B. Huang)", (None, None)),
])
def test_label_from_comment(comment, expected):
    assert label_from_comment(comment) == expected


def _loop(brief, n=50, op=True):
    d = {"Comments": np.array(["line one", "line two"], dtype=object), "BriefComments": brief,
         "Type": 0, "Ts": 1, "t": np.arange(n), "SP": np.zeros(n), "PV": np.sin(np.arange(n))}
    d["OP"] = np.cos(np.arange(n)) if op else np.zeros(3)      # wrong length -> NaN
    return d


def test_load_isdb(tmp_path):
    f = tmp_path / "isdb10.mat"
    sio.savemat(f, {"cdata": {
        "chemicals": {"loop10": _loop("Pressure control; with stiction (B. Huang)"),
                      "loop12": _loop("Flow control; with stiction (B. Huang)", op=False)},
        "testdata": {"loop1": _loop("synthetic")}}})
    loops = {x.key: x for x in load_isdb(f)}
    assert set(loops) == {"chemicals.10", "chemicals.12"}          # testdata excluded
    a = loops["chemicals.10"]
    assert (a.loop_type, a.contributor, a.label, a.tier) == ("P", "B. Huang", "stiction", "stated")
    assert a.in_sacac == "stiction-P-chemical-baccidicapaci-2018"
    assert loops["chemicals.12"].data.op.isna().all()
    assert len(load_isdb(f, include_test=True)) == 3
