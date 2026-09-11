"""SACAC loader - each test is one quirk actually found in the repository.
Synthetic files only: the data is not committed."""
import pytest

from chemai.data import load_record, load_sacac, read_loop_csv
from chemai.data.sacac import parse_name


def _write(path, header, rows, sep=";", eol="\r"):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [sep.join(header)] + [sep.join(str(v) for v in r) for r in rows]
    path.write_bytes(eol.join(lines).encode())
    return path


def test_reads_by_name_not_position(tmp_path):
    """Three files put PV before SP."""
    f = _write(tmp_path / "unknown-P-oilgas-thornhill-2007-1.csv",
               ["Time", "PV", "SP", "OP", "Error"], [[1, 5.0, 3.0, 40.0, -2.0]])
    df = read_loop_csv(f)
    assert df.loc[0, "sp"] == 3.0 and df.loc[0, "pv"] == 5.0 and "error" in df


def test_bare_cr_line_endings_and_case(tmp_path):
    f = _write(tmp_path / "stiction-F-paper-horch-2003.csv", ["time", "SP", "PV", "OP"],
               [[i, 110, 107 + i, 14] for i in range(5)], eol="\r")
    assert len(read_loop_csv(f)) == 5


def test_missing_signals_become_nan(tmp_path):
    f = _write(tmp_path / "quantisation-T-chemicals-Thornhill-2003.csv", ["Time", "PV"],
               [[1, 2.0], [2, 2.1]])
    df = read_loop_csv(f)
    assert df.sp.isna().all() and df.op.isna().all() and df.pv.notna().all()


def test_db_runs_share_one_loop():
    a = parse_name("stiction-P-oilgas-DB-3-baccidicapaci-2018")
    b = parse_name("stiction-P-oilgas-DB-7-baccidicapaci-2018")
    c = parse_name("unknown-P-oilgas-thornhill-2007-1")
    d = parse_name("unknown-P-oilgas-thornhill-2007-2")
    assert a["loop_id"] == b["loop_id"]
    assert c["loop_id"] != d["loop_id"]                 # PC1 and PC2: two loops
    assert a["loop_type"] == "P" and a["source"] == "baccidicapaci"


def test_copied_rows_are_cut_not_edited(tmp_path):
    f = _write(tmp_path / "tuning-L-paper-horch-2003.csv", ["time", "SP", "PV", "OP"],
               [[i, 50, 50, 58] for i in range(1147)])
    rec = load_record(f)
    assert len(rec.data) == 846 and rec.rows_dropped == 301
    assert len(read_loop_csv(f)) == 1147               # raw file untouched


@pytest.mark.parametrize("stem,label", [
    ("tuning-F-chemical-DB-4-baccidicapaci-2018", "tuning_sluggish"),
    ("tuning-Q-paper-horch-2003", "tuning_tight"),
    ("other-F-paper-horch-2003-2", "healthy"),           # exact name beats prefix
    ("other-F-paper-horch-2003", "external_oscillation"),
    ("stiction-L-power-baccidicapaci-2018", "stiction"),
    ("unknown-F-paper-horch-2003", None),
    ("other-F-chemicals-thornhill-2003", None),
])
def test_labels_follow_published_descriptions(tmp_path, stem, label):
    f = _write(tmp_path / f"{stem}.csv", ["time", "SP", "PV", "OP"], [[1, 1, 1, 1], [2, 1, 1, 1]])
    assert load_record(f).label == label


def test_known_dead_time_attached(tmp_path):
    f = _write(tmp_path / "other-L-paper-horch-2003.csv", ["time", "SP", "PV", "OP"],
               [[2 * i, 1, 1, 1] for i in range(10)])
    (tmp_path / "other-L-paper-horch-2003.txt").write_text("Sampling rate: 2 seconds\n")
    rec = load_record(f)
    assert rec.dead_time == (4.0, 4.0) and rec.ts == 2.0


def test_sampling_time_undefined_when_sources_disagree(tmp_path):
    """stiction-L-paper-horch: description says 1 s, time column steps 2."""
    f = _write(tmp_path / "stiction-L-paper-horch-2003.csv", ["time", "SP", "PV", "OP"],
               [[2 * i, 1, 1, 1] for i in range(10)])
    (tmp_path / "stiction-L-paper-horch-2003.txt").write_text("Sampling rate: 1 second\n")
    rec = load_record(f)
    assert rec.ts_meta == 1.0 and rec.ts_column == 2.0 and rec.ts is None


def test_non_constant_sampling_gives_no_rate(tmp_path):
    f = _write(tmp_path / "stiction-P-oilgas-DB-1-baccidicapaci-2018.csv",
               ["Time", "SP", "PV", "OP"], [["18/07/24 0:35", 1, 1, 1]])
    (tmp_path / "d.txt").write_text("Sampling rate: non-constant (mean value: 12 seconds)\n")
    assert load_record(f).ts_meta is None


def test_plantwide_folder_skipped(tmp_path):
    _write(tmp_path / "Stiction" / "stiction-F-paper-horch-2003.csv", ["time", "SP", "PV", "OP"],
           [[1, 1, 1, 1]])
    _write(tmp_path / "Plantwide Data" / "x" / "plantwide-minerals-brooks.csv", ["Tagname", "LC1"],
           [[1, 2]], sep=",")
    recs = load_sacac(tmp_path)
    assert [r.name for r in recs] == ["stiction-F-paper-horch-2003"]


def test_irregular_sampling_gives_no_ts(tmp_path):
    f = _write(tmp_path / "tuning-Q-paper-horch-2003.csv", ["time", "SP", "PV", "OP"],
               [[t, 1, 1, 1] for t in (1, 2, 4, 5, 8)])
    (tmp_path / "tuning-Q-paper-horch-2003.txt").write_text("Sampling rate: 1 second\n")
    assert load_record(f).ts is None
