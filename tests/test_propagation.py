"""Plant-wide propagation: clustering, band share, and what the ranking cannot do."""
import numpy as np
import pytest
from scipy.signal import lfilter

from chemai.features import (band_share, dominant_periods, find_oscillation_cluster,
                             rank_source_candidates)

FS, N, PERIOD = 1.0, 3000, 40.0
t = np.arange(N)
rng = np.random.default_rng(0)


def _plant(square_source=False):
    """One source and two victims at the same period, plus one unrelated loop.
    `square_source` makes the source non-linear, as a sticking valve is."""
    wave = np.sign(np.sin(2 * np.pi * t / PERIOD)) if square_source else np.sin(2 * np.pi * t / PERIOD)
    source = wave + 0.1 * rng.normal(size=N)
    def victim(tau, noise, seed):
        a = np.exp(-1 / tau)
        return lfilter([1 - a], [1, -a], source) + noise * rng.normal(size=N)
    other = np.sin(2 * np.pi * t / 7) + 0.5 * np.random.default_rng(9).normal(size=N)
    return (np.column_stack([source, victim(3, 0.15, 1), victim(8, 0.3, 2), other]),
            ["source", "victim_near", "victim_far", "unrelated"])


def test_dominant_periods():
    signals, _ = _plant()
    periods = dominant_periods(signals, FS)
    assert np.allclose(periods[:3], PERIOD, rtol=0.1)
    assert abs(periods[3] - 7) < 1                      # the unrelated loop keeps its own


def test_band_share_falls_with_distance_from_the_source():
    signals, _ = _plant()
    shares = band_share(signals, PERIOD, FS)
    assert shares[0] > shares[1] > shares[2] > shares[3]


def test_cluster_groups_the_three_and_excludes_the_fourth():
    event = find_oscillation_cluster(*_plant(), fs=FS)
    assert event is not None
    assert set(event.members) == {"source", "victim_near", "victim_far"}
    assert event.period == pytest.approx(PERIOD, rel=0.1)


def test_no_cluster_when_nothing_is_shared():
    signals = np.column_stack([np.sin(2 * np.pi * t / p) + rng.normal(size=N)
                               for p in (11.0, 29.0, 57.0)])
    assert find_oscillation_cluster(signals, ["a", "b", "c"], fs=FS) is None


def test_ranking_marks_one_candidate_and_the_rest_victims():
    table = rank_source_candidates(find_oscillation_cluster(*_plant(), fs=FS))
    assert table.iloc[0].loop == "source" and table.iloc[0].role == "source candidate"
    assert (table.iloc[1:].role == "likely victim").all()
    assert "No work order" in table.iloc[1].action


def test_victims_are_everything_but_the_leading_candidate():
    event = find_oscillation_cluster(*_plant(), fs=FS)
    assert "source" not in event.victims() and len(event.victims()) == len(event.members) - 1


def test_the_event_carries_its_own_caveat():
    """The ranking is a filter, not a verdict - the report must say so."""
    event = find_oscillation_cluster(*_plant(), fs=FS)
    assert "not proof" in event.note and "field" in event.note.lower()


def test_a_clean_victim_can_outrank_a_noisy_source():
    """Why the ranking is only a filter: band share measures nearness to the
    source, and a quiet victim beats a source sitting in a noisy loop."""
    source = np.sin(2 * np.pi * t / PERIOD) + 1.5 * rng.normal(size=N)   # noisy loop
    a = np.exp(-1 / 2)
    quiet_victim = lfilter([1 - a], [1, -a], np.sin(2 * np.pi * t / PERIOD))
    shares = band_share(np.column_stack([source, quiet_victim]), PERIOD, FS)
    assert shares[1] > shares[0]


def test_a_nonlinear_source_is_penalised_by_its_own_fingerprint():
    """A sticking valve makes a square-ish limit cycle, whose power is spread
    into harmonics - so LESS of it sits in the fundamental band, and a victim
    that received a filtered (purer) version can outrank it. This is the
    quantitative explanation of the horch trio (VALIDATION.md 20.4)."""
    signals, names = _plant(square_source=True)
    shares = band_share(signals, PERIOD, FS)
    assert shares[1] > shares[0]                      # the near victim outranks the source


def test_input_checks():
    signals, names = _plant()
    with pytest.raises(ValueError):
        find_oscillation_cluster(signals, names[:2], fs=FS)
    with pytest.raises(ValueError):
        dominant_periods(signals[:, 0], FS)
