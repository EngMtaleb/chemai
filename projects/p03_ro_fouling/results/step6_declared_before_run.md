# Declared before the external test was run

Written after steps 4 (development plants) and 5 (test plants), before step 6
touched the forecasts on the real deposit.

- Frozen: linear law, 42-day trailing window, 3 flushing days skipped, interval
  inflation 2.0 (P10-P90), flag beyond 2x the observed window, plan a CIP when
  the P50 date is within 28 days. Review levels scored: 1.15 and 1.20.
- Primary analysis: the linear law. Secondary, reported but not adopted: the gap
  law (DP_norm^(-1/2.2)), which was 1-2 days better on simulated plants - a gain
  that may be built in, because the simulator itself uses the gap law.
- Expectation: the deposit's DP_norm was normalised by its authors with a
  vendor-style flow exponent (1.5, per operating mode). On simulated plants a
  vendor exponent cost about 4 days of median error and let a simple margin rule
  beat the forecast. If the real plant's exponent is nearer the physics (1.7-1.8),
  the same may happen here. A forecast that does not beat the margin rule on the
  deposit will be reported as such.
- Unit of every score: the cleaning cycle. Days inside a cycle are not
  independent; the cycle bootstrap gives the intervals.
