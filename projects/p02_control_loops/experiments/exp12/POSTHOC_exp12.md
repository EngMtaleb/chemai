# Experiment 12 — one check after the results (post hoc)

Written: 2026-10-25, AFTER the experiment-12 network results were seen. It changes nothing in
PROTOCOL_exp12.md and adds no method to the comparison with the rule.

## Why

The rule-as-teacher networks differ from the experiment-11 networks in two ways at once: they have a
two-class output (sticky or not), trained on every labelled real loop of the four training parts, AND
they learn from the 11 rule-labelled loops. A change against the experiment-11 network cannot say which
of the two caused it.

## The check

The same two-class CNN and LSTM, trained exactly like the teacher networks (same windows, weights, seeds,
early stopping, condition B folds) — but WITHOUT the 11 rule-labelled loops. The difference between the
teacher and this check is what the 11 loops added.

## Reading

Reported next to the teacher networks as "two-class, without the 11 loops (checked after the run)".
It is not a new try against the bar.
