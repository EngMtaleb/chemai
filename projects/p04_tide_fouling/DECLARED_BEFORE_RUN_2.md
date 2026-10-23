# Project 4, part 2 — the sun, declared before the tests were run

Written and committed after the tide test (part 1) and before either test below
was fitted to the real data. Same rules as part 1: a change after this commit
is listed in VALIDATION.md with its reason, and both versions are reported.

## 1. The questions

A plant run on solar power produces in daylight and stops at night. Two things
the record can answer decide what that costs the pre-treatment membrane:

- **H2 — the sun's clock.** Does the fouling rate follow the time of day, with
  one phase across 2012–2015? If so, the hours a solar plant runs (09:00–15:00)
  foul faster or slower than the rest of the day, and by how much.
- **H3 — the morning restart.** In the first 3 hours after the unit has stood
  idle for at least 2 hours, does it foul at a different rate than in the rest
  of its first 12 hours? A solar plant restarts every morning.

## 2. What was done before this declaration

- Part 1 (the tide): declared, run, reported (commits `1989858`, `18466f4`).
- Power for both tests (step5, `results/step5_sun_power.json`), on synthetic
  effects in noise shaped like the data; neither real effect was estimated.
  - H2: a daily wave whose 5–95 % range is 10 % of the median fouling rate is
    found at p ≤ 0.01 in 99.5 % of 200 trials; 5 % in 40.5 %.
  - H3 (2,503 segments, 81 restarts): an early-hours step of 20 % of the median
    is found in 98 % of 300 trials; 10 % in 58 %; 5 % in 12 %.

## 3. H2 — the test

- **Response**: fouling rate per segment (`fouling_pct_h`), as in part 1.
- **Base model**: run levels; per-episode slow trend (part 1's spline);
  ln(flow) as a deviation from the run mean; ln(segment minutes); element-pair
  dummies. **No time-of-day block** (it is what is tested). **No temperature
  term**: viscosity is already inside R, and a scheduler needs the whole effect
  of the hour, whatever carries it. Sensitivity run: with temperature.
- **Tested block**: cos/sin of the first four harmonics of the day (24, 12, 8,
  6 h) on the logger's local standard clock.
- **Null**: 999 surrogates; each shift block's clock is moved by an independent
  uniform 1–23 h. That scrambles the time of day and nothing else.
- **p** = (1 + surrogates at or above the observed statistic) / 1000.
- **Positive control (C3)**: the same test on the segment mean temperature,
  which is known to peak in the afternoon, must give p ≤ 0.01.
- **Placebo**: periods 11.0, 13.5, 27.0 and 5.3 h, same base model and null,
  on the fouling rate, must give p > 0.01.
- **Confirmation**: daily component fitted on runs before 2014-01-01, frozen,
  must remove a positive share of the residual variance of runs from
  2014-01-01 on, with p ≤ 0.05 against the same hour-shift null.
- **Decision number — the solar-hours contrast**: mean of the fitted daily
  component over 09:00–15:00 minus its mean over the other 18 hours, as a share
  of the median fouling rate. 95 % interval from 999 bootstrap resamples of
  shift blocks (the final regression only; the base model is not refitted).
  **Relevant** if its absolute value is at least 10 % of the median.

## 4. H3 — the test

- **Data**: segments of episodes that began after the unit stood idle ≥ 2 h,
  within the first 12 h of the episode.
- **Model**: fouling rate ~ episode level + early (first 3 h) + time of day
  (4 harmonics of 24 h) + ln(flow) deviation + ln(segment minutes) + element
  pair. Within-episode winsorising at 1 % / 99 %.
- **Test**: the `early` coefficient, CR1 cluster-robust standard error by
  episode, two-sided t with (episodes − 1) degrees of freedom. Passes at
  p ≤ 0.01.
- **Placebo**: the same model with a pretend restart 24 h into episodes of
  36 h or more must give p > 0.01.
- **Confirmation**: the early coefficient has the same sign in the 2012–2013
  restarts and in the 2014–2015 restarts, with p ≤ 0.05 in each.
- **Relevant** if |early coefficient| ≥ 10 % of the median fouling rate.

## 5. What the outcomes will mean

| outcome | meaning for a solar-run plant |
|---|---|
| H2 passes, confirmed, controls hold | the hour matters; the contrast says whether solar hours foul faster or slower, and the operating plan uses the measured daily profile |
| H2 fails | no daily pattern larger than about 10 % of the median; the plan treats every hour alike for fouling |
| H3 passes and replicates | the morning restart has a price (faster) or a bonus (slower); the plan counts it once per day |
| H3 fails | no restart effect larger than about 20 % of the median (98 % power); daily stop-start carries no detectable fouling price |
| C3 or a placebo fails | the test is not trusted; the matching H is reported as uninterpretable |

H2 and H3 are each tested at 0.01; the chance of a false pass on either is
under 2 %.

## 6. Reported, not decisive

- H2 with temperature in the base model.
- The fitted daily profile hour by hour.
- H3 split by idle time (2–12 h against more than 12 h) — exploratory.

## 7. What this is not

One pilot. A daily pattern can come from the sun, from the water's own daily
cycle, or from the site's daily routine (shift changes, sampling, manual
adjustments); the test cannot tell them apart, and a routine would not carry
over to an unattended solar plant. Restarts in the record were the operators'
choice, not random: an episode that began after a cleaning or a repair may
start cleaner for that reason.
