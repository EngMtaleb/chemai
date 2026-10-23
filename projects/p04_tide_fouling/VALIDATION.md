# Project 4 — validation notes

Every claim in README.md traces to a section here, and every number here to a
file in `results/` or a test in `tests/`.

## 1. The record, as found

Public UF pilot record, Zenodo 4630769 (CC0; also Dryad doi:10.5068/D1310B),
md5 `317ba90d029d86fd7d69208c53b3c4d6`. An ultrafiltration unit pre-treating
seawater for a reverse-osmosis pilot. 180 run files in the depositors'
`training/` (150) and `test/` (30) folders. Source: `results/step1_audit.json`.

| finding | count | what was done |
|---|---|---|
| csv with a logged clock (Excel day number) | 149 + 1 without header | used |
| files with no clock (sheets, or csv saved from sheets) | 30 | not usable against a clock |
| clocked files whose first sample is not on the day their name says | 6 (one a year off, one with day and month swapped) | excluded, not guessed |
| malformed lines (39 fields where 18 belong) | 102 lines in 5 files | skipped and counted, never repaired |
| clocked runs shorter than 6 h | 37 | excluded |
| **kept** | **107 runs, 5,105.8 h, 2012-06-23 → 2015-04-01** | |

- **The clock is local civil time.** In all 34 kept runs of 48 h or more, the
  daily temperature maximum falls between 12 and 19 h (median hour 16). A clock
  logged in UTC would put it near midnight.
- **Turbidity is often not feed turbidity.** The tag reads −2.5 (no signal) in
  63 kept runs and about 0.03 NTU (filtrate) in 25. Only 19 runs, all in 2013,
  read the feed (median ≥ 0.20 NTU): 1,517.9 h.
- **The depositors' split is not separated in time.** Three `test/` files
  continue a `training/` run without a break (episodes found by chaining files
  whose start follows the previous end within 30 min). A model scored on that
  split has seen the minutes just before its test data.

## 2. Filtration segments

Source: `results/step2_segments.json`; code `chemai/features/uf_segments.py`;
tests `tests/test_uf_segments.py`.

- Three elements. In 2012 all three filter together and a backwash stops them
  all; from March 2013 two filter while the third is backwashed and rested, and
  the pair changes every 18–28 minutes.
- A segment is a stretch of steady filtration (total flow within ±15 % of the
  run median, TMP > 0.5 psi) with one set of elements online, a change counting
  only after it holds 60 s. The first 180 s (switching and backwash-pulse
  transients) and the last 60 s are dropped; at least 10 minutes and 5 binned
  points must remain.
- Fouling rate = slope of ln R against time on 1-minute medians (2-minute
  medians where the logger sampled every 120 s), R = TMP/(μ(T)·Q). The
  viscosity term removes the TMP swing a temperature change causes; the test
  `test_fouling_slope_is_recovered_and_not_fooled_by_temperature` checks it.
- 14,998 segments in 106 runs (one run never filtered steadily), 86 episodes.
  Fouling rate median 13.2 %/h, interquartile 8.7–18.6 %/h, 3.7 % negative.
  A single segment's slope is noisy: its standard error is a median 37 % of its
  size. The test works on thousands of segments, not on any one.
