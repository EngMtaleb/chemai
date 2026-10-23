# Declared before the run — Part D: when to clean an RO train, by energy alone

Written on 2026-10-23, **before any energy, slope or interval was computed** for the plant's
cleaning cycles. Committed before `step4_ro_cip_economics.py` is run; nothing above the line at
the end is edited after the run.

## The question

A seawater RO train fouls; to hold its production it needs more net driving pressure and its
feed-channel pressure drop rises, so every cubic metre costs a little more energy than the last.
A cleaning (CIP) resets that, at a fixed cost. **If energy were the only reason to clean, when
would the cleanings fall — and how does that compare with when the operators cleaned?**

The rule of Part A, at a time scale of weeks: clean when the live energy penalty of the next
cubic metre reaches the cycle's average cost so far, cleaning included. Under linear growth it
is the square-root law, T* = √(2F/a).

## The data

The RO plant deposit of Project 3 (Su, Meng & Chang 2026; doi:10.5281/zenodo.21055948,
CC BY 4.0): three trains, daily indicators over two years, 18 cleaning campaigns. Used here:
the **14 complete cycles** (a cleaning at each end, 18–148 days). The three initial stretches
(no cleaning before them, long gaps) and three open ones (3–7 days) are left out.

What was looked at before writing this: the cycle list and lengths, and the share of zero
`ndp_pct` days per cycle (15–65 %; the indicator is floored at its post-cleaning baseline). No
slope, penalty or cost was computed.

## The cost model

Per cubic metre of permeate, at day t of a cycle:

    live cost  m(t) = price × kWh_per_bar × [ NDP0 × ndp_pct(t) + ½ × DP0 × (dp_pct(t) − dp_start) ]

- `ndp_pct`: the rise of the net driving pressure since the last cleaning (it resets at each
  cleaning, P3 VALIDATION). `dp_pct`: the normalised pressure drop against a fixed
  commissioning baseline; `dp_start` = its median over days 3–10 of the cycle.
- kWh_per_bar = [1 + (1 − η_ERD)(1 − r)/r] / (36 η_pump): **0.0368** with an isobaric energy-
  recovery device (η_pump 0.80, η_ERD 0.95, recovery r 0.45), **0.0772** without one.
- The deposit withholds absolute pressures and prices, so the plant constants are assumptions,
  run on a grid stated now.
- Ending a cycle costs **F**, in dollars per m³/day of the train's capacity: chemicals, labour,
  flushing and lost production. F = 1 is, for example, $1,000 for a train making 1,000 m³/day,
  or one day of lost water valued at $1/m³.

| | Mid case | Corner most in favour of cleaning | Grid |
|---|---|---|---|
| Electricity, $/kWh | 0.10 | 0.15 | 0.05, 0.10, 0.15 |
| Clean net driving pressure NDP0, bar | 20 | 25 | 15, 20, 25 |
| Clean feed-channel pressure drop DP0, bar | 2 | 3 | 1, 2, 3 |
| Energy recovery | yes | no | both |
| Cost of a cleaning F, $ per m³/day | 1.0 | 0.5 | 0.5, 1, 2, 5 |

## What is computed per cycle (and nothing else)

The first 3 days after a cleaning (flushing) are dropped. Then:

- **the rule**: the first day, from day 7 on, when the trailing 7-day median of m(t) reaches the
  average cost J(t) = [F + ∫m dt] / t. *None* if it never does before the operators' cleaning;
- **the law**: a = Theil–Sen slope of m(t) against t; T* = √(2F/a);
- **the implied cost**: F_implied = a·T² / 2 — what a cleaning would have to cost for the
  interval the operators chose to be the energy-optimal one;
- for reference, the day the 7-day median of DP_norm first reached the published 1.20 trigger.

## Predictions, written before the run

**D1.** Mid case: the rule reaches its stop before the operators' cleaning in **at most 4 of the
14** cycles. (Expectation: the energy penalty grows too slowly for weeks-long cycles to pay for a
cleaning.)

**D2.** Mid case: the median implied cleaning cost is **below $0.25 per m³/day** — a quarter of
the mid-case F, and below any realistic cleaning.

**D3.** Even in the corner most in favour of cleaning, the rule stops before the operators in
**at most 7 of the 14** cycles.

**D4** (reported, no pass mark): Spearman correlation between a cycle's length and the pressure
drop its closing cleaning did not remove (the post-cleaning median of `dp_pct` minus the cycle's
`dp_start`). A clear positive value would be a hidden cost of waiting that energy alone does not
see.

Also reported: the same count for every one of the 216 grid cases, and T*/T_actual per cycle.

## What would count against the expectation

D1 or D2 failing means energy alone would have justified most of these cleanings — the
operators' timing would then be close to the energy optimum, and the README says so. Passing
means something other than energy set these cleanings — the pressure-drop limit that protects the
elements, salt passage, or a fixed routine — and that a cleaning schedule defended by energy
savings is defended by the smallest of its reasons. It would not mean the plant cleaned too often:
a constraint is not a cost, and Project 3 is about forecasting exactly that limit.

---
*Corrections after the run (dated, never edits above this line):*

- 2026-10-23, after the run: the deposit has **15** complete cycles, not 14 — train C has six,
  not five; the count above was made by hand from the cycle list and is wrong by one. The
  thresholds were set for 14 cycles; the results (D1: 0 of 15, D3: 6 of 15) clear them either way.
  The first run also printed the T*/T_actual quantiles as NaN because a cycle with no rise has an
  infinite T*; the summary now reports the share of such cycles and the quantiles of the others.
  No number used by D1–D4 changed.
