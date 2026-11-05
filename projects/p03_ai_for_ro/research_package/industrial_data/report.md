# Industrial RO — independently reproduced derived-indicator forecast

## Evidence and provenance

Downloaded Zenodo version DOI 10.5281/zenodo.21055948, verified the published ZIP MD5 and every entry in SHA256SUMS.txt. These are real-plant-derived daily indicator records, not invented data and not raw SCADA.

Rows: 1340; trains: {'A': 381, 'B': 485, 'C': 474}; relative days 0–771; CIP campaigns: 18. No exact calendar dates, raw sensor pressures, membrane dimensions, or chemical-water-quality channels are supplied.

## Source dictionary / semantic limit

| Field | Meaning used in this study | Excluded interpretation |
|---|---|---|
| dp_pct | Deposited dimensionless pressure-related indicator | Not measured ΔP in bar; not verified DP_norm−1 |
| flux_pct | Deposited dimensionless flux-related indicator | Not measured flow m³/h |
| ndp_pct, sp_pct | Additional current and causal historical indicator inputs | Not reconstructed feed pressure / product salinity |
| day_index | Relative chronology and split only | Not absolute weather-aligned date |
| days_to_next_cip, state | Future-aware metadata excluded from predictors | Not an available future sensor |
| CIP ledger | Past event history and retrospective pair exclusions | Not a prescriptive cleaning policy |

Deposit dictionary says fixed-baseline deviations, while deposited scripts distinguish per-CIP dp_pct/ndp_pct from fixed-baseline DP_norm. Keep exact deposited labels; do not assign raw pressure or energy semantics. Upstream derivation cannot be audited without withheld raw data.

The deposited source scripts (head2head.py lines distinguishing RAW and PERCIP, angle1_trigger.py description) explicitly separate fixed-baseline DP_norm from per-CIP dp_pct/ndp_pct. The dictionary names *_pct fixed-baseline deviations. As upstream raw data are withheld, the discrepancy cannot be resolved here. We preserve the exact names and make no physical ΔP threshold or energy conversion.

## Method

Plan frozen before fitting: study_plan.json. Four fixed methods; future-minus-current regression for Ridge and HistGradientBoosting. Models receive current values and strictly trailing history of four deposited indicators. Historical windows reset after known cleaning. All preprocessing is fitted to training rows only. Missing future observations are not interpolated. No train ID, absolute day index, future CIP interval, or event-window state is a model feature.

Validation: origin ≥450 and target <550; development training target <450. Final rolling tests: origin days 550–604, 605–659, 660–714, 715–771. Each fit only uses target days before its test-block start. Separate generalization test holds out an entire train and fits the other two trains before day550.

The pair exclusion uses the future CIP ledger retrospectively only to define a conditional no-intervening-CIP benchmark. This is not an unconditional operational forecast: the future cleaning schedule would not generally be known at deployment. Overlapping horizons and autocorrelated daily records are not independent samples. No confidence intervals or universal reliability claim are made.

## Results (MAE in deposited dimensionless units; lower is better)

Persistence has the lowest validation MAE in all eight target/horizon combinations. The later 14-day flux smoothing gain is a test observation and research candidate; it was not the best validation-selected method. A fair operational default remains the simple baseline pending new prospective evidence.

| protocol | target | horizon | method | n | mae | rmse | skill |
| --- | --- | --- | --- | --- | --- | --- | --- |
| held_out_train | dp_pct | 1 | persistence | 633 | 0.01733 | 0.02381 | 0.00000 |
| held_out_train | dp_pct | 1 | median3_7d | 633 | 0.02105 | 0.02787 | -0.21492 |
| held_out_train | dp_pct | 1 | ridge_residual | 633 | 0.03701 | 0.04821 | -1.13576 |
| held_out_train | dp_pct | 1 | histgb_residual | 633 | 0.02205 | 0.03032 | -0.27250 |
| held_out_train | dp_pct | 3 | persistence | 609 | 0.02808 | 0.03709 | 0.00000 |
| held_out_train | dp_pct | 3 | median3_7d | 609 | 0.02744 | 0.03631 | 0.02284 |
| held_out_train | dp_pct | 3 | ridge_residual | 609 | 0.09714 | 0.13753 | -2.45893 |
| held_out_train | dp_pct | 3 | histgb_residual | 609 | 0.03422 | 0.04922 | -0.21833 |
| held_out_train | dp_pct | 7 | persistence | 566 | 0.02889 | 0.04007 | 0.00000 |
| held_out_train | dp_pct | 7 | median3_7d | 566 | 0.02835 | 0.03819 | 0.01885 |
| held_out_train | dp_pct | 7 | ridge_residual | 566 | 0.09655 | 0.13975 | -2.34191 |
| held_out_train | dp_pct | 7 | histgb_residual | 566 | 0.04396 | 0.05790 | -0.52162 |
| held_out_train | dp_pct | 14 | persistence | 496 | 0.03499 | 0.04531 | 0.00000 |
| held_out_train | dp_pct | 14 | median3_7d | 496 | 0.03373 | 0.04354 | 0.03595 |
| held_out_train | dp_pct | 14 | ridge_residual | 496 | 0.06943 | 0.09452 | -0.98431 |
| held_out_train | dp_pct | 14 | histgb_residual | 496 | 0.04005 | 0.05047 | -0.14473 |
| held_out_train | flux_pct | 1 | persistence | 633 | 0.02103 | 0.04059 | 0.00000 |
| held_out_train | flux_pct | 1 | median3_7d | 633 | 0.02442 | 0.04452 | -0.16116 |
| held_out_train | flux_pct | 1 | ridge_residual | 633 | 0.02620 | 0.04142 | -0.24588 |
| held_out_train | flux_pct | 1 | histgb_residual | 633 | 0.02453 | 0.04009 | -0.16655 |
| held_out_train | flux_pct | 3 | persistence | 609 | 0.03060 | 0.05237 | 0.00000 |
| held_out_train | flux_pct | 3 | median3_7d | 609 | 0.03015 | 0.05096 | 0.01457 |
| held_out_train | flux_pct | 3 | ridge_residual | 609 | 0.04443 | 0.06827 | -0.45229 |
| held_out_train | flux_pct | 3 | histgb_residual | 609 | 0.03834 | 0.05258 | -0.25309 |
| held_out_train | flux_pct | 7 | persistence | 566 | 0.03498 | 0.05884 | 0.00000 |
| held_out_train | flux_pct | 7 | median3_7d | 566 | 0.03404 | 0.05766 | 0.02692 |
| held_out_train | flux_pct | 7 | ridge_residual | 566 | 0.06263 | 0.09138 | -0.79032 |
| held_out_train | flux_pct | 7 | histgb_residual | 566 | 0.05266 | 0.06630 | -0.50511 |
| held_out_train | flux_pct | 14 | persistence | 496 | 0.04351 | 0.06725 | 0.00000 |
| held_out_train | flux_pct | 14 | median3_7d | 496 | 0.04043 | 0.06343 | 0.07091 |
| held_out_train | flux_pct | 14 | ridge_residual | 496 | 0.06321 | 0.08365 | -0.45260 |
| held_out_train | flux_pct | 14 | histgb_residual | 496 | 0.04399 | 0.06624 | -0.01094 |
| rolling | dp_pct | 1 | persistence | 633 | 0.01733 | 0.02381 | 0.00000 |
| rolling | dp_pct | 1 | median3_7d | 633 | 0.02105 | 0.02787 | -0.21492 |
| rolling | dp_pct | 1 | ridge_residual | 633 | 0.02469 | 0.03315 | -0.42466 |
| rolling | dp_pct | 1 | histgb_residual | 633 | 0.01947 | 0.02666 | -0.12346 |
| rolling | dp_pct | 3 | persistence | 609 | 0.02808 | 0.03709 | 0.00000 |
| rolling | dp_pct | 3 | median3_7d | 609 | 0.02744 | 0.03631 | 0.02284 |
| rolling | dp_pct | 3 | ridge_residual | 609 | 0.05043 | 0.06443 | -0.79575 |
| rolling | dp_pct | 3 | histgb_residual | 609 | 0.03133 | 0.04173 | -0.11549 |
| rolling | dp_pct | 7 | persistence | 566 | 0.02889 | 0.04007 | 0.00000 |
| rolling | dp_pct | 7 | median3_7d | 566 | 0.02835 | 0.03819 | 0.01885 |
| rolling | dp_pct | 7 | ridge_residual | 566 | 0.05002 | 0.06428 | -0.73127 |
| rolling | dp_pct | 7 | histgb_residual | 566 | 0.03043 | 0.04092 | -0.05316 |
| rolling | dp_pct | 14 | persistence | 496 | 0.03499 | 0.04531 | 0.00000 |
| rolling | dp_pct | 14 | median3_7d | 496 | 0.03373 | 0.04354 | 0.03595 |
| rolling | dp_pct | 14 | ridge_residual | 496 | 0.03830 | 0.05136 | -0.09454 |
| rolling | dp_pct | 14 | histgb_residual | 496 | 0.03366 | 0.04394 | 0.03786 |
| rolling | flux_pct | 1 | persistence | 633 | 0.02103 | 0.04059 | 0.00000 |
| rolling | flux_pct | 1 | median3_7d | 633 | 0.02442 | 0.04452 | -0.16116 |
| rolling | flux_pct | 1 | ridge_residual | 633 | 0.02293 | 0.03758 | -0.09060 |
| rolling | flux_pct | 1 | histgb_residual | 633 | 0.02478 | 0.04013 | -0.17861 |
| rolling | flux_pct | 3 | persistence | 609 | 0.03060 | 0.05237 | 0.00000 |
| rolling | flux_pct | 3 | median3_7d | 609 | 0.03015 | 0.05096 | 0.01457 |
| rolling | flux_pct | 3 | ridge_residual | 609 | 0.02965 | 0.04529 | 0.03092 |
| rolling | flux_pct | 3 | histgb_residual | 609 | 0.03489 | 0.05022 | -0.14023 |
| rolling | flux_pct | 7 | persistence | 566 | 0.03498 | 0.05884 | 0.00000 |
| rolling | flux_pct | 7 | median3_7d | 566 | 0.03404 | 0.05766 | 0.02692 |
| rolling | flux_pct | 7 | ridge_residual | 566 | 0.03574 | 0.05223 | -0.02153 |
| rolling | flux_pct | 7 | histgb_residual | 566 | 0.04688 | 0.06223 | -0.34009 |
| rolling | flux_pct | 14 | persistence | 496 | 0.04351 | 0.06725 | 0.00000 |
| rolling | flux_pct | 14 | median3_7d | 496 | 0.04043 | 0.06343 | 0.07091 |
| rolling | flux_pct | 14 | ridge_residual | 496 | 0.05052 | 0.06878 | -0.16105 |
| rolling | flux_pct | 14 | histgb_residual | 496 | 0.03871 | 0.05781 | 0.11039 |
| validation | dp_pct | 1 | persistence | 230 | 0.01243 | 0.01922 | 0.00000 |
| validation | dp_pct | 1 | median3_7d | 230 | 0.01502 | 0.02360 | -0.20829 |
| validation | dp_pct | 1 | ridge_residual | 230 | 0.04057 | 0.05316 | -2.26446 |
| validation | dp_pct | 1 | histgb_residual | 230 | 0.03005 | 0.04267 | -1.41827 |
| validation | dp_pct | 3 | persistence | 212 | 0.01935 | 0.02824 | 0.00000 |
| validation | dp_pct | 3 | median3_7d | 212 | 0.02158 | 0.03133 | -0.11489 |
| validation | dp_pct | 3 | ridge_residual | 212 | 0.14296 | 0.17220 | -6.38696 |
| validation | dp_pct | 3 | histgb_residual | 212 | 0.05985 | 0.08862 | -2.09274 |
| validation | dp_pct | 7 | persistence | 182 | 0.03063 | 0.04433 | 0.00000 |
| validation | dp_pct | 7 | median3_7d | 182 | 0.03318 | 0.04761 | -0.08329 |
| validation | dp_pct | 7 | ridge_residual | 182 | 0.12956 | 0.16464 | -3.22973 |
| validation | dp_pct | 7 | histgb_residual | 182 | 0.06058 | 0.08166 | -0.97765 |
| validation | dp_pct | 14 | persistence | 133 | 0.04740 | 0.06390 | 0.00000 |
| validation | dp_pct | 14 | median3_7d | 133 | 0.05027 | 0.06738 | -0.06049 |
| validation | dp_pct | 14 | ridge_residual | 133 | 0.09567 | 0.14391 | -1.01826 |
| validation | dp_pct | 14 | histgb_residual | 133 | 0.05052 | 0.06814 | -0.06572 |
| validation | flux_pct | 1 | persistence | 230 | 0.01387 | 0.03504 | 0.00000 |
| validation | flux_pct | 1 | median3_7d | 230 | 0.01530 | 0.03728 | -0.10269 |
| validation | flux_pct | 1 | ridge_residual | 230 | 0.02045 | 0.03803 | -0.47424 |
| validation | flux_pct | 1 | histgb_residual | 230 | 0.02452 | 0.03995 | -0.76760 |
| validation | flux_pct | 3 | persistence | 212 | 0.02092 | 0.04895 | 0.00000 |
| validation | flux_pct | 3 | median3_7d | 212 | 0.02121 | 0.05025 | -0.01380 |
| validation | flux_pct | 3 | ridge_residual | 212 | 0.03715 | 0.06158 | -0.77581 |
| validation | flux_pct | 3 | histgb_residual | 212 | 0.04007 | 0.05908 | -0.91585 |
| validation | flux_pct | 7 | persistence | 182 | 0.02955 | 0.06716 | 0.00000 |
| validation | flux_pct | 7 | median3_7d | 182 | 0.02988 | 0.06787 | -0.01095 |
| validation | flux_pct | 7 | ridge_residual | 182 | 0.08835 | 0.11620 | -1.98948 |
| validation | flux_pct | 7 | histgb_residual | 182 | 0.06032 | 0.08191 | -1.04099 |
| validation | flux_pct | 14 | persistence | 133 | 0.05321 | 0.10105 | 0.00000 |
| validation | flux_pct | 14 | median3_7d | 133 | 0.05771 | 0.10708 | -0.08467 |
| validation | flux_pct | 14 | ridge_residual | 133 | 0.06653 | 0.11444 | -0.25037 |
| validation | flux_pct | 14 | histgb_residual | 133 | 0.06274 | 0.10456 | -0.17915 |

## Predefined research screen

Required: ≥5% MAE improvement over persistence on pooled rolling tests AND positive skill on at least3/4 time blocks AND ≥5% pooled held-out-train improvement. This is a research screen, not operational acceptance.

| target | horizon | method | rolling_skill | held_out_train_skill | positive_blocks | passes |
| --- | --- | --- | --- | --- | --- | --- |
| dp_pct | 1 | median3_7d | -0.21492 | -0.21492 | 1 | False |
| dp_pct | 1 | ridge_residual | -0.42466 | -1.13576 | 0 | False |
| dp_pct | 1 | histgb_residual | -0.12346 | -0.27250 | 1 | False |
| dp_pct | 3 | median3_7d | 0.02284 | 0.02284 | 3 | False |
| dp_pct | 3 | ridge_residual | -0.79575 | -2.45893 | 0 | False |
| dp_pct | 3 | histgb_residual | -0.11549 | -0.21833 | 1 | False |
| dp_pct | 7 | median3_7d | 0.01885 | 0.01885 | 3 | False |
| dp_pct | 7 | ridge_residual | -0.73127 | -2.34191 | 0 | False |
| dp_pct | 7 | histgb_residual | -0.05316 | -0.52162 | 1 | False |
| dp_pct | 14 | median3_7d | 0.03595 | 0.03595 | 3 | False |
| dp_pct | 14 | ridge_residual | -0.09454 | -0.98431 | 1 | False |
| dp_pct | 14 | histgb_residual | 0.03786 | -0.14473 | 1 | False |
| flux_pct | 1 | median3_7d | -0.16116 | -0.16116 | 0 | False |
| flux_pct | 1 | ridge_residual | -0.09060 | -0.24588 | 0 | False |
| flux_pct | 1 | histgb_residual | -0.17861 | -0.16655 | 1 | False |
| flux_pct | 3 | median3_7d | 0.01457 | 0.01457 | 3 | False |
| flux_pct | 3 | ridge_residual | 0.03092 | -0.45229 | 3 | False |
| flux_pct | 3 | histgb_residual | -0.14023 | -0.25309 | 1 | False |
| flux_pct | 7 | median3_7d | 0.02692 | 0.02692 | 2 | False |
| flux_pct | 7 | ridge_residual | -0.02153 | -0.79032 | 1 | False |
| flux_pct | 7 | histgb_residual | -0.34009 | -0.50511 | 1 | False |
| flux_pct | 14 | median3_7d | 0.07091 | 0.07091 | 3 | True |
| flux_pct | 14 | ridge_residual | -0.16105 | -0.45260 | 2 | False |
| flux_pct | 14 | histgb_residual | 0.11039 | -0.01094 | 2 | False |

## Reproducibility and limits

A future-mutation test changes all future channel values, removes all future CIP events, and corrupts future-aware metadata; it verifies exact invariance of every pre-cutoff feature row. The deposit also contains18 train-days with n_10min_samples>144, which requires upstream clarification; no counts are changed or invented, and their sum is not presented as independent raw records.

run_study.py regenerates forecast_pairs.csv, predictions.csv, metrics.csv, metrics_by_split.csv, results.json, plot_data.json and three charts from the verified deposited CSVs. Full pair/split audit is in results.json. Training code is new, so this is an independent reanalysis inspired by the attachment, not a claim to have rerun its unavailable full notebooks.

Public source data license: CC BY4.0; original source code: MIT. Attribution: Yi Hsiang Su, Fan Cheng Meng and Pieh Yu Chang, Auditable Clean-in-Place Decision Support from Routine SWRO SCADA, Membranes2026,16(8),257; DOI10.3390/membranes16080257. Dataset depositor: Yi Hsiang Su. Exact version DOI10.5281/zenodo.21055948.

No result validates autonomous CIP decisions, industrial electricity savings, plant-wide fault labels, or generalization to other facilities. For stronger field evidence: raw calibrated sensors, feed-water quality, exact CIP actions, operation modes and prospective evaluation on another plant are needed.