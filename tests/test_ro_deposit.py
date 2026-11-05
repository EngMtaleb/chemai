"""The deposit loader - tested on a small table built to look like the deposit,
so no download is needed. Each test is a rule found by reading the real files."""
import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from chemai.config import RODepositConfig
from chemai.data.ro_deposit import (audit, cleaning_legs, download_deposit, group_campaigns,
                                    load_daily, load_ledger, recovery_by_campaign, split_cycles)


def _table(tmp_path):
    """Two trains. Train A cleaned on day 40 (one leg); train B on days 30 and 32
    (acid then alkaline: two legs, one campaign) and on day 70."""
    rows = []
    for train, cips in (("A", [40]), ("B", [30, 32, 70])):
        for d in range(0, 100):
            if d in cips:
                continue
            prev = [c for c in cips if c <= d]
            nxt = [c for c in cips if c > d]
            dp = 0.0 if d < 5 else 0.002 * (d - (max(prev) if prev else 0))
            rows.append(dict(train=train, day_index=d, n_10min_samples=150 if d == 50 else 72,
                             basis="n3", ndp_pct=0.0, dp_pct=dp, flux_pct=0.0, sp_pct=0.0,
                             days_since_cip=(d - max(prev)) if prev else np.nan,
                             days_to_next_cip=(min(nxt) - d) if nxt else np.nan, state="mid"))
    daily = tmp_path / "derived_daily_indicator_table.csv"
    pd.DataFrame(rows).to_csv(daily, index=False)
    ledger = tmp_path / "cip_campaign_ledger.csv"
    pd.DataFrame([dict(train="A", episode="A#1", day_index=40, stage="acid"),
                  dict(train="B", episode="B#1", day_index=30, stage="acid+alkaline"),
                  dict(train="B", episode="B#2", day_index=70, stage="alkaline")]).to_csv(ledger, index=False)
    return load_daily(daily), load_ledger(ledger)


def test_legs_are_read_back_from_the_table_and_collapse_into_campaigns(tmp_path):
    daily, ledger = _table(tmp_path)
    legs = cleaning_legs(daily)
    assert legs == {"A": [40], "B": [30, 32, 70]}
    assert group_campaigns(legs["B"], 7) == [(30, 32), (70, 70)]
    assert audit(daily, ledger)["campaigns_match_ledger"]


def test_cycles_are_cut_at_cleanings_and_never_span_one(tmp_path):
    daily, _ = _table(tmp_path)
    cycles = split_cycles(daily)
    b = [c for c in cycles if c.train == "B"]
    assert [(c.start, c.end, c.kind) for c in b] == [(0, 30, "initial"), (32, 70, "complete"),
                                                       (70, 100, "open")]
    for c in cycles:                                   # no day inside a cycle crosses a cleaning
        assert np.all(c.abs_day > c.start) or c.kind == "initial"
        assert np.all(c.abs_day < c.end)


def test_a_zero_is_censored_not_clean(tmp_path):
    daily, _ = _table(tmp_path)
    zeros = daily[daily["dp_pct"] == 0]
    assert zeros["censored"].all() and (zeros["dp_norm"] == 1.0).all()


def test_more_than_144_ten_minute_samples_in_a_day_is_flagged(tmp_path):
    daily, ledger = _table(tmp_path)
    assert daily["over_max"].sum() == 2                # day 50, both trains
    assert audit(daily, ledger)["over_max_days"] == 2


def test_recovery_needs_three_days_each_side(tmp_path):
    daily, ledger = _table(tmp_path)
    rec = recovery_by_campaign(daily, ledger)
    assert rec["evaluable"].all()
    assert (rec["drop"] > 0).all()                     # every cleaning lowered dP in this table


def test_download_refuses_a_file_with_the_wrong_md5(tmp_path):
    cfg = RODepositConfig()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("data/derived_daily_indicator_table.csv", "not the deposit")
    (tmp_path / cfg.zip_name).write_bytes(buf.getvalue())
    with pytest.raises(ValueError, match="md5"):
        download_deposit(tmp_path, cfg)


def test_missing_deposit_says_how_to_get_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_deposit"):
        load_daily(tmp_path / "nope.csv")
