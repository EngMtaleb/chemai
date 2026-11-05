# Phase 4 — Fouling's energy on a plant that runs on the sun, and whether the sun changes when to clean: declared before any number is computed

Written 29 October 2026, before any fouled-train energy, fouled year or cleaning interval was computed. Nothing here will be
changed after seeing a result; any later change is listed as a change, with its reason, next to the original.

## Why

The merged RO project has two goals. Each has its answer so far:

- **Fouling forecast.** With the routine daily indicators of the public plant, no method beat *no change* for ΔP, 1–14 days
  ahead (phase 3; Project 3 and Project 6 agree). A simulator that would have shown the value of logging the causes could not be
  made to behave like the plant (phase 3b, stopped).
- **Energy.** The phase-2 plant, twice its demand in size with 0.8 kWp of panels per m³/day and a 12-hour tank, runs about four
  fifths on the sun with no forecast. And on this plant, energy alone paid for an RO cleaning before the operators cleaned in
  0 of 15 cycles in Project 5's mid case, and in at most 6 of 15 across its 216 price-and-plant cases (Project 5, Part D).

Phase 4 joins them on one train: **how much energy does this plant's fouling cost, and does running on the sun change when
energy alone would clean?**

## Inputs — all from earlier work, nothing new is fitted

- **The train and the year (phase 2):** seven FilmTec SW30HRLE-440 elements fitted to the datasheet, 43 g/L, design flux
  14 LMH, 40 % recovery (35 % where the hot sea needs it), every FilmTec limit; per 1,000 m³/day of demand, 8 trains; the rest of
  the plant at 65/35; sun and sea of 2024 from NASA POWER; the **steady** strategy (normal plant, demand every hour) and the
  **rule** strategy (the example design: 2×, 0.8 kWp per m³/day, 12-hour tank).
- **The fouling (the public deposit):** per train-day, days 300–771: `dp_pct` (ΔP rise on a fixed commissioning baseline) and
  `flux_pct` (loss of 25 °C specific flux, fixed baseline, floored at 0); for the cleaning question, Project 5's inputs exactly:
  the 15 complete cycles, `ndp_pct` (the net driving pressure's rise since the last cleaning — this indicator is reset at each
  cleaning, as Project 3 found) and the ΔP rise above the cycle's start.
- **The cleaning economics (Project 5, Part D):** its live cost, rule and square-root law, its first-3-days skip, 7-day smoothing
  and 7-day floor; its price grid 0.05, 0.10, 0.15 USD/kWh and cleaning cost F 0.5, 1, 2, 5 USD per m³/day of capacity.

## Fouling on the model train

Every element's feed-channel pressure drop × (1 + d); the membrane's water permeability × (1 − f); salt permeability unchanged.
d and f come from the deposit: this plant's fouling, put on our train — a what-if, since the deposit gives no absolute pressures.

## What is computed (and nothing else)

1. **The penalty map.** At the design point (14 LMH): feed pressure, RO energy and the limits, for d = 0, 0.1, 0.2, 0.3, 0.4 ×
   f = 0, 0.05, 0.10, 0.20, at 15, 25 and 35 °C; the extra RO energy over the clean train, in kWh/m³ and in % of it.
2. **This plant's fouling on our train.** Every train-day's (d, f), days 300–771, at 25 °C (a fine map, interpolated): the
   median and the 90th percentile of the extra RO energy.
3. **A year under the sun, fouled.** 2024, with the trains held all year at one state: clean; the plant's median day
   (d and f at their medians); the level its operators cleaned at (the fingerprint's 22.8 % with that level's median f); both at
   their 90th percentiles. Holding a state all year is an upper bound for it, not a schedule. Per strategy (steady, rule): total
   and grid kWh/m³, solar share, and the **grid share of the fouling penalty** = extra grid kWh ÷ extra total kWh.
4. **When would energy alone clean, with the sun?** Project 5's rule and law on the 15 cycles, with the phase-2 train's own
   constants instead of Project 5's assumed ones — the clean net driving pressure (14 LMH ÷ the membrane's permeability at 25 °C),
   the clean vessel's pressure drop and the energy per bar of feed pressure, from the model — and the price multiplied by the grid
   share of item 3 (the sun pays the rest). For each price × F and each strategy: the cycles in which energy alone would have
   cleaned before the operators did.

## Written before the run — what is expected

- **E1.** On our train at 25 °C, the plant's median fouling day costs less than 1 % of the clean RO energy.
- **E2.** Under the rule strategy, the grid carries a smaller share of the fouling penalty than under the steady strategy.
- **E3.** With the sun, energy alone cleans before the operators in no more cycles than without it, at every price and F: the sun
  does not make energy a reason to clean.
- **E4.** At the 90th-percentile state, no element passes 1 bar of pressure drop at the design point at 25 °C: within this plant's
  fouling range, the element pressure-drop limit is not what fouling breaks first at the design point.

## Checks

- With d = f = 0, the code reproduces phase 2 exactly: the design-point table and the 2024 rule and steady results.
- Item 4 with Project 5's own constants (no sun) reproduces Project 5's mid-case count.

## Reported whatever the result

Every map value, the four states, both strategies, every price × F, and E1–E4 with pass or fail. The notebook re-runs it from
the public deposit and the public NASA POWER files.

## Changes after the first run (29 October 2026)

- **The recovery plan is held when the train fouls.** The declaration did not say whether a fouled train keeps the clean
  plant's recovery plan (40 %, and 35 % at 35 °C, where the clean design point breaks the lead element's limits). The first run
  recomputed the plan for each fouled state, and stopped at the 90th-percentile state: there the fouled membrane — its flux
  spread more evenly along the vessel — passes the lead element's limits at 40 % even at 35 °C, so no temperature needed 35 %,
  and the code had no case for an empty list. Every fouled state now keeps the clean plant's plan, so the penalty measures
  fouling alone; the penalty map (item 1) already did. The two states that had run, the median day and the operators' cleaning
  level, kept 35 % at 35 °C either way, so their results do not change. The recomputed plan is reported as a sensitivity for
  the 90th-percentile state; it is not used in item 4 or in E1–E4.
- **E3 is checked for both strategies.** The run script first checked the rule strategy only; the declaration says "with the
  sun", which covers both. Both are now checked.
- The penalty map and the fine map were computed once, in the first run; the second run read them back (they do not depend on
  the change above).
- **E1 has two readings.** E1 speaks of "the plant's median fouling day". The run script, written before the run, scored it as
  the median of the train-days' costs (item 2); item 3 calls the median state — the median ΔP rise with the median
  permeability loss — "the plant's median day". The notebook reports both: 1.11 % and 0.90 %.
- **Which grid share prices item 4.** The declaration says "the grid share of item 3"; the run script, written before the run,
  takes the largest of the three fouled states (the one most in favour of cleaning). The notebook also shows each state's own
  share and what it gives.
- **The notebook's check of phase 2 is lighter than the run script's:** the design points, the operating table's size and the
  2024 rule result. The run script also matched phase 2's ten-year steady and rule averages.
- **Added after the run, descriptive only:** section 5a (each train's operating window by sea temperature for three states, the
  2024 split of sun and grid energy, and the first element's flux and recovery at the window's edges), the cost of the median
  and 90th-percentile states read from the 25 °C map, and the full table of the penalty map.
