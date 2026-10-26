# Declared before the run — Parts B and C: when to backwash and when to clean a UF membrane

Written on 2026-10-23, **before any backwash count, cost or interval was computed**. Committed
before `step5_uf_economics.py` is run; nothing above the line at the end is edited afterwards.

## The question

The same rule as Parts A and D, on the ultrafiltration pre-treatment of Project 4 (UF pilot record,
doi:10.5068/D1310B, CC0, 2012–2015):

- **B — backwash, minutes.** A backwash spends filtrate and pump energy. Between backwashes the
  transmembrane pressure (TMP) climbs, so each cubic metre costs a little more energy to push
  through. If energy and water were the only costs, how much should be filtered per backwash —
  and how does that compare with what the pilot filtered?
- **C — chemical cleaning, months.** What the backwashes leave behind accumulates; the resistance
  after backwash climbs until a chemical clean resets it. If the extra energy were the only cost
  of waiting, when would the cleans fall?

## What was looked at before writing this

The segment table of Project 4 (all years) and the daily median element resistance it gives,
including its large day-to-day falls (2013-05-30, 2013-10-09, 2014-03-18, 2014-07-28, and one
across the Oct 2014 – Jan 2015 gap). On the development years only (2012–2013), a first regression
of the fouling left after each rotation against the filtration period: its sign flipped between two
reasonable specifications, so **no claim about how the period length changes irreversible fouling
is made in Project 5** — it is reported as open (VALIDATION.md), with the experiment that would
settle it. No backwash count, backwash volume per event, cost or interval was computed.

## B — the backwash economics (per run)

Runs with at least 3 backwashes, more than 1 m³ filtered and a positive median fouling rate.

- A **backwash** is backwash flow above 5 gpm; pulses less than 120 s apart are one backwash;
  events shorter than 20 s are noise.
- Fixed cost of a backwash: F = (backwash volume × value of the filtrate it uses) + (its pump
  energy, hydraulic ∫p·q dt, ÷ pump efficiency × electricity price).
- Growth of the live cost per m³ filtered: a = price / (36 η) × TMP × (fouling rate) / flow —
  the run's median TMP, median filtration flow and median segment fouling rate (Project 4).
- Energy-only best volume per backwash: **V\* = √(2F/a)**; the pilot's volume per backwash:
  V_cycle = volume filtered / number of backwashes. Reported: V\*/V_cycle.

| | Mid case | Corner most in favour of frequent backwash | Grid |
|---|---|---|---|
| Electricity, $/kWh | 0.10 | 0.15 | 0.05, 0.10, 0.15 |
| Value of a m³ of filtrate used to backwash, $ | 0.05 | 0.01 | 0.01, 0.05, 0.10 |
| Pump efficiency η | 0.7 | 0.5 | 0.5, 0.7 |

**B1.** Mid case: the median V\*/V_cycle over runs is **at least 5** — energy and water alone would
have backwashed at least five times less often than the pilot did.

**B2.** In the corner most in favour of backwashing, the median is still **at least 2**.

Expectation behind both: a run's TMP rises by a few percent between backwashes, worth a fraction
of a cent per m³, while each backwash spends about a quarter of the volume filtered before it.

## C — chemical cleaning (per interval between cleans)

- Daily baseline: the daily median, over filtration segments, of the per-element resistance at a
  segment's start (ln R of the unit + ln of the elements online); days with at least 5 segments.
- A **chemical clean**: a fall of more than 0.4 in that ln R from the previous logged day, the two
  days at most 30 days apart (a fall across a longer gap cannot be dated).
- Live cost at day t after a clean: the energy of the extra TMP the risen resistance needs at the
  clean-state flow: m(t) = price / (36 η) × TMP_clean × (e^(ln R(t) − ln R_clean) − 1), with
  TMP_clean and ln R_clean the medians of days 1–7.
- Per interval between two consecutive cleans: the rule's day (trailing 7-day median of m reaching
  the average cost, from day 7), the slope a, T\* = √(2F/a), and the implied cost F_implied = aT²/2.
- F, the cost of a chemical clean, in $ per m³/day of UF capacity: **0.3** in the mid case
  (grid 0.05, 0.1, 0.3, 1.0), with electricity 0.10 $/kWh and η 0.7 (grid as in B).

**C** is reported without a pass mark: T\*/T_actual and the implied cost for each interval, and
the range of the median over the grid. Expectation: unlike the RO train of Part D, the UF
resistance roughly doubles between cleans, so energy alone should come within a factor of three
of the actual intervals — but this is a guess, not a test.

## What would count against the expectation

B1 or B2 failing would mean the pilot's backwash frequency was close to what energy and water
alone justify. Passing would mean the backwash interval was set by something else — keeping the
TMP below a limit, and the fouling a longer filtration might leave behind, which this record
cannot price (above) — and that the pilot's backwash water, roughly a quarter of its feed, is the
cost to look at first.

---
*Corrections after the run (dated, never edits above this line):*
