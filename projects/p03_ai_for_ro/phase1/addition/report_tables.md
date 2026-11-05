### Overview

#### ΔP (`dp_pct`) — rolling origin

*No change* MAE, percentage points: 1 d 1.76, 3 d 2.83, 7 d 2.91, 14 d 3.49

| method | 1 d: skill % (blocks > 0) | bar | 3 d: skill % (blocks > 0) | bar | 7 d: skill % (blocks > 0) | bar | 14 d: skill % (blocks > 0) | bar |
|---|---|---|---|---|---|---|---|---|
| smoothed no change | -20.8 (1/4) | fail | +2.2 (3/4) | fail | +1.8 (2/4) | fail | +3.6 (4/4) | fail |
| trend | -4.5 (0/4) | fail | -12.5 (0/4) | fail | -26.4 (0/4) | fail | -53.2 (0/4) | fail |
| fouling law | -48.5 (0/4) | fail | -6.3 (1/4) | fail | -16.2 (1/4) | fail | -20.4 (1/4) | fail |
| XGBoost | -15.7 (1/4) | fail | -0.8 (1/4) | fail | -9.3 (2/4) | fail | -14.6 (1/4) | fail |
| hybrid | -50.4 (0/4) | fail | -36.6 (1/4) | fail | -24.9 (1/4) | fail | -32.2 (0/4) | fail |
| **Prophet** (new) | -309.4 (0/4) | fail | -165.7 (0/4) | fail | -171.6 (0/4) | fail | -151.0 (0/4) | fail |
| **Prophet + level shift** (new) | -91.9 (0/4) | fail | -27.6 (1/4) | fail | -31.7 (1/4) | fail | -26.9 (0/4) | fail |
| **LSTM** (new) | -6.4 (1/4) | fail | -11.8 (0/4) | fail | -6.5 (2/4) | fail | -10.9 (1/4) | fail |
| **RNN** (new) | -1.6 (2/4) | fail | -4.9 (1/4) | fail | -11.8 (3/4) | fail | -17.5 (1/4) | fail |

#### flux loss (`flux_pct`) — rolling origin

*No change* MAE, percentage points: 1 d 2.11, 3 d 3.07, 7 d 3.48, 14 d 4.31

| method | 1 d: skill % (blocks > 0) | bar | 3 d: skill % (blocks > 0) | bar | 7 d: skill % (blocks > 0) | bar | 14 d: skill % (blocks > 0) | bar |
|---|---|---|---|---|---|---|---|---|
| smoothed no change | -15.9 (0/4) | fail | +1.4 (2/4) | fail | +2.7 (2/4) | fail | +7.1 (3/4) | **PASS** |
| trend | -3.1 (0/4) | fail | -8.9 (0/4) | fail | -22.1 (0/4) | fail | -27.3 (0/4) | fail |
| fouling law | -44.9 (0/4) | fail | -13.6 (1/4) | fail | -7.9 (2/4) | fail | +1.0 (1/4) | fail |
| XGBoost | -12.1 (1/4) | fail | -2.7 (2/4) | fail | -5.8 (2/4) | fail | +3.6 (2/4) | fail |
| hybrid | -28.7 (0/4) | fail | -16.9 (2/4) | fail | -18.7 (2/4) | fail | -3.3 (2/4) | fail |
| **Prophet** (new) | -91.4 (0/4) | fail | -33.4 (1/4) | fail | -21.8 (1/4) | fail | -3.6 (2/4) | fail |
| **Prophet + level shift** (new) | -54.4 (0/4) | fail | -10.4 (2/4) | fail | -1.8 (2/4) | fail | +10.0 (2/4) | fail |
| **LSTM** (new) | -5.9 (0/4) | fail | -2.5 (2/4) | fail | -10.3 (1/4) | fail | -16.9 (1/4) | fail |
| **RNN** (new) | -5.9 (0/4) | fail | -4.8 (1/4) | fail | -10.7 (1/4) | fail | -11.8 (1/4) | fail |

#### ΔP (`dp_pct`) — unseen train

*No change* MAE, percentage points: 1 d 1.76, 3 d 2.83, 7 d 2.91, 14 d 3.49

| method | 1 d: skill % (blocks > 0) | bar | 3 d: skill % (blocks > 0) | bar | 7 d: skill % (blocks > 0) | bar | 14 d: skill % (blocks > 0) | bar |
|---|---|---|---|---|---|---|---|---|
| smoothed no change | -20.8 (1/4) | fail | +2.2 (3/4) | fail | +1.8 (2/4) | fail | +3.6 (4/4) | fail |
| trend | -4.5 (0/4) | fail | -12.5 (0/4) | fail | -26.4 (0/4) | fail | -53.2 (0/4) | fail |
| fouling law | -48.5 (0/4) | fail | -6.3 (1/4) | fail | -16.2 (1/4) | fail | -20.4 (1/4) | fail |
| XGBoost | -22.9 (0/4) | fail | -12.4 (1/4) | fail | -28.5 (1/4) | fail | -24.6 (1/4) | fail |
| hybrid | -78.0 (0/4) | fail | -58.7 (1/4) | fail | -67.3 (1/4) | fail | -57.9 (0/4) | fail |
| **Prophet** (new) | -309.4 (0/4) | fail | -165.7 (0/4) | fail | -171.6 (0/4) | fail | -151.0 (0/4) | fail |
| **Prophet + level shift** (new) | -91.9 (0/4) | fail | -27.6 (1/4) | fail | -31.7 (1/4) | fail | -26.9 (0/4) | fail |
| **LSTM** (new) | -3.0 (0/4) | fail | -9.7 (0/4) | fail | -32.0 (0/4) | fail | -22.6 (1/4) | fail |
| **RNN** (new) | -4.7 (0/4) | fail | -14.7 (0/4) | fail | -11.4 (1/4) | fail | -23.4 (1/4) | fail |

#### flux loss (`flux_pct`) — unseen train

*No change* MAE, percentage points: 1 d 2.11, 3 d 3.07, 7 d 3.48, 14 d 4.31

| method | 1 d: skill % (blocks > 0) | bar | 3 d: skill % (blocks > 0) | bar | 7 d: skill % (blocks > 0) | bar | 14 d: skill % (blocks > 0) | bar |
|---|---|---|---|---|---|---|---|---|
| smoothed no change | -15.9 (0/4) | fail | +1.4 (2/4) | fail | +2.7 (2/4) | fail | +7.1 (3/4) | **PASS** |
| trend | -3.1 (0/4) | fail | -8.9 (0/4) | fail | -22.1 (0/4) | fail | -27.3 (0/4) | fail |
| fouling law | -44.9 (0/4) | fail | -13.6 (1/4) | fail | -7.9 (2/4) | fail | +1.0 (1/4) | fail |
| XGBoost | -18.3 (0/4) | fail | -13.6 (1/4) | fail | -17.9 (2/4) | fail | +1.7 (2/4) | fail |
| hybrid | -41.0 (0/4) | fail | -27.0 (1/4) | fail | -31.1 (1/4) | fail | -6.3 (1/4) | fail |
| **Prophet** (new) | -91.4 (0/4) | fail | -33.4 (1/4) | fail | -21.8 (1/4) | fail | -3.6 (2/4) | fail |
| **Prophet + level shift** (new) | -54.4 (0/4) | fail | -10.4 (2/4) | fail | -1.8 (2/4) | fail | +10.0 (2/4) | fail |
| **LSTM** (new) | -6.4 (0/4) | fail | -11.6 (1/4) | fail | -23.9 (0/4) | fail | -24.2 (0/4) | fail |
| **RNN** (new) | -6.4 (0/4) | fail | -9.1 (0/4) | fail | -24.7 (0/4) | fail | -20.4 (0/4) | fail |

### Every horizon in detail

**ΔP, 1 d ahead, rolling origin** (n = 642)

| method | MAE, pp | skill % | block 1 | block 2 | block 3 | block 4 | R² of the change | bar |
|---|---|---|---|---|---|---|---|---|
| no change | 1.76 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | -0.00 | — |
| smoothed no change | 2.13 | -20.8 | -27.7 | -52.0 | -14.1 | +1.7 | -0.35 | fail |
| trend | 1.84 | -4.5 | -6.6 | -5.2 | -7.1 | -1.1 | -0.09 | fail |
| fouling law | 2.62 | -48.5 | -68.1 | -50.0 | -62.2 | -27.3 | -1.01 | fail |
| XGBoost | 2.04 | -15.7 | -20.6 | -6.8 | -53.0 | +5.8 | -0.44 | fail |
| hybrid | 2.65 | -50.4 | -76.2 | -68.9 | -73.5 | -6.7 | -1.06 | fail |
| **Prophet** (new) | 7.23 | -309.4 | -690.5 | -159.4 | -297.1 | -219.0 | -13.57 | fail |
| **Prophet + level shift** (new) | 3.39 | -91.9 | -120.3 | -79.9 | -148.6 | -46.7 | -2.18 | fail |
| **LSTM** (new) | 1.88 | -6.4 | -5.0 | -4.2 | +2.2 | -14.6 | -0.12 | fail |
| **RNN** (new) | 1.79 | -1.6 | -9.6 | -3.1 | +3.9 | +0.2 | -0.03 | fail |

**ΔP, 3 d ahead, rolling origin** (n = 617)

| method | MAE, pp | skill % | block 1 | block 2 | block 3 | block 4 | R² of the change | bar |
|---|---|---|---|---|---|---|---|---|
| no change | 2.83 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | -0.00 | — |
| smoothed no change | 2.76 | +2.2 | +8.7 | -5.6 | +0.9 | +8.9 | 0.04 | fail |
| trend | 3.18 | -12.5 | -14.2 | -12.3 | -19.3 | -4.4 | -0.26 | fail |
| fouling law | 3.00 | -6.3 | -13.3 | +12.5 | -15.9 | -17.1 | -0.10 | fail |
| XGBoost | 2.85 | -0.8 | -3.8 | +6.4 | -8.5 | -0.6 | -0.01 | fail |
| hybrid | 3.86 | -36.6 | -27.7 | +4.7 | -42.0 | -99.2 | -0.90 | fail |
| **Prophet** (new) | 7.51 | -165.7 | -346.7 | -27.7 | -171.3 | -190.4 | -5.66 | fail |
| **Prophet + level shift** (new) | 3.61 | -27.6 | -36.2 | +9.1 | -67.7 | -32.6 | -0.49 | fail |
| **LSTM** (new) | 3.16 | -11.8 | -13.3 | -0.7 | -29.8 | -8.4 | -0.20 | fail |
| **RNN** (new) | 2.96 | -4.9 | -14.9 | +1.5 | -6.1 | -3.8 | -0.07 | fail |

**ΔP, 7 d ahead, rolling origin** (n = 572)

| method | MAE, pp | skill % | block 1 | block 2 | block 3 | block 4 | R² of the change | bar |
|---|---|---|---|---|---|---|---|---|
| no change | 2.91 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | -0.02 | — |
| smoothed no change | 2.86 | +1.8 | -3.8 | +5.0 | -2.8 | +6.0 | 0.07 | fail |
| trend | 3.68 | -26.4 | -45.2 | -29.6 | -31.7 | -4.7 | -0.61 | fail |
| fouling law | 3.39 | -16.2 | -76.5 | +9.7 | -10.9 | -25.9 | -0.17 | fail |
| XGBoost | 3.18 | -9.3 | -42.6 | +2.2 | +1.4 | -19.9 | -0.12 | fail |
| hybrid | 3.64 | -24.9 | -111.6 | +4.5 | -24.2 | -18.8 | -0.32 | fail |
| **Prophet** (new) | 7.91 | -171.6 | -583.9 | -29.7 | -150.6 | -166.6 | -5.31 | fail |
| **Prophet + level shift** (new) | 3.84 | -31.7 | -123.8 | +12.8 | -49.1 | -23.2 | -0.46 | fail |
| **LSTM** (new) | 3.10 | -6.5 | -59.2 | +0.4 | +9.5 | -5.3 | -0.08 | fail |
| **RNN** (new) | 3.26 | -11.8 | -101.9 | +0.6 | +5.7 | +1.3 | -0.14 | fail |

**ΔP, 14 d ahead, rolling origin** (n = 502)

| method | MAE, pp | skill % | block 1 | block 2 | block 3 | block 4 | R² of the change | bar |
|---|---|---|---|---|---|---|---|---|
| no change | 3.49 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | -0.07 | — |
| smoothed no change | 3.36 | +3.6 | +9.9 | +1.5 | +2.4 | +4.0 | 0.02 | fail |
| trend | 5.34 | -53.2 | -72.5 | -79.3 | -42.1 | -18.3 | -1.52 | fail |
| fouling law | 4.20 | -20.4 | -91.8 | -6.7 | +5.8 | -34.2 | -0.43 | fail |
| XGBoost | 4.00 | -14.6 | -77.0 | -7.9 | +9.4 | -20.7 | -0.43 | fail |
| hybrid | 4.61 | -32.2 | -93.2 | -32.8 | -7.7 | -28.6 | -1.12 | fail |
| **Prophet** (new) | 8.75 | -151.0 | -404.4 | -57.1 | -134.8 | -131.7 | -5.43 | fail |
| **Prophet + level shift** (new) | 4.43 | -26.9 | -78.6 | -16.1 | -18.1 | -20.1 | -0.58 | fail |
| **LSTM** (new) | 3.87 | -10.9 | -53.0 | -20.2 | +22.9 | -25.3 | -0.21 | fail |
| **RNN** (new) | 4.10 | -17.5 | -72.5 | -18.7 | +20.4 | -41.8 | -0.35 | fail |

**flux loss, 1 d ahead, rolling origin** (n = 642)

| method | MAE, pp | skill % | block 1 | block 2 | block 3 | block 4 | R² of the change | bar |
|---|---|---|---|---|---|---|---|---|
| no change | 2.11 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | -0.00 | — |
| smoothed no change | 2.44 | -15.9 | -24.9 | -42.8 | -8.0 | -8.7 | -0.20 | fail |
| trend | 2.17 | -3.1 | -3.8 | -4.5 | -2.6 | -2.8 | -0.05 | fail |
| fouling law | 3.05 | -44.9 | -34.6 | -79.0 | -45.0 | -29.7 | -0.32 | fail |
| XGBoost | 2.36 | -12.1 | -72.2 | -20.1 | -15.0 | +4.9 | 0.08 | fail |
| hybrid | 2.71 | -28.7 | -110.0 | -41.6 | -26.0 | -11.4 | -0.11 | fail |
| **Prophet** (new) | 4.03 | -91.4 | -274.7 | -68.5 | -58.6 | -103.6 | -0.80 | fail |
| **Prophet + level shift** (new) | 3.25 | -54.4 | -52.1 | -60.1 | -61.7 | -44.9 | -0.52 | fail |
| **LSTM** (new) | 2.23 | -5.9 | -29.6 | -5.4 | -1.9 | -6.0 | 0.05 | fail |
| **RNN** (new) | 2.23 | -5.9 | -47.3 | -5.4 | -2.5 | -2.4 | 0.07 | fail |

**flux loss, 3 d ahead, rolling origin** (n = 617)

| method | MAE, pp | skill % | block 1 | block 2 | block 3 | block 4 | R² of the change | bar |
|---|---|---|---|---|---|---|---|---|
| no change | 3.07 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | -0.00 | — |
| smoothed no change | 3.02 | +1.4 | +10.0 | -7.4 | +8.3 | -0.1 | 0.05 | fail |
| trend | 3.34 | -8.9 | -4.5 | -12.0 | -9.0 | -7.5 | -0.16 | fail |
| fouling law | 3.48 | -13.6 | +3.3 | -15.5 | -26.6 | -3.6 | 0.00 | fail |
| XGBoost | 3.15 | -2.7 | -81.4 | +13.0 | -8.3 | +5.9 | 0.13 | fail |
| hybrid | 3.58 | -16.9 | -100.2 | +1.4 | -37.6 | +4.5 | -0.09 | fail |
| **Prophet** (new) | 4.09 | -33.4 | -144.6 | +11.7 | -35.7 | -42.8 | -0.12 | fail |
| **Prophet + level shift** (new) | 3.39 | -10.4 | +9.3 | +12.9 | -39.2 | -4.5 | 0.01 | fail |
| **LSTM** (new) | 3.14 | -2.5 | -46.8 | +4.5 | -9.7 | +7.3 | 0.12 | fail |
| **RNN** (new) | 3.21 | -4.8 | -60.0 | -1.9 | -6.0 | +4.5 | 0.14 | fail |

**flux loss, 7 d ahead, rolling origin** (n = 572)

| method | MAE, pp | skill % | block 1 | block 2 | block 3 | block 4 | R² of the change | bar |
|---|---|---|---|---|---|---|---|---|
| no change | 3.48 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | -0.00 | — |
| smoothed no change | 3.39 | +2.7 | -13.3 | +8.0 | -5.3 | +7.7 | 0.04 | fail |
| trend | 4.25 | -22.1 | -27.0 | -17.5 | -26.8 | -21.0 | -0.40 | fail |
| fouling law | 3.75 | -7.9 | -28.7 | +1.9 | -35.8 | +13.4 | 0.06 | fail |
| XGBoost | 3.68 | -5.8 | -196.2 | +19.0 | -22.4 | +11.0 | 0.15 | fail |
| hybrid | 4.13 | -18.7 | -237.8 | +20.0 | -60.0 | +13.6 | -0.10 | fail |
| **Prophet** (new) | 4.24 | -21.8 | -250.6 | +22.9 | -39.7 | -17.2 | 0.03 | fail |
| **Prophet + level shift** (new) | 3.54 | -1.8 | -31.1 | +25.7 | -44.5 | +19.1 | 0.15 | fail |
| **LSTM** (new) | 3.84 | -10.3 | -155.9 | +14.3 | -18.8 | -6.6 | 0.12 | fail |
| **RNN** (new) | 3.85 | -10.7 | -162.0 | +1.1 | -7.0 | -6.8 | 0.14 | fail |

**flux loss, 14 d ahead, rolling origin** (n = 502)

| method | MAE, pp | skill % | block 1 | block 2 | block 3 | block 4 | R² of the change | bar |
|---|---|---|---|---|---|---|---|---|
| no change | 4.31 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | -0.00 | — |
| smoothed no change | 4.00 | +7.1 | -0.1 | +16.4 | +3.5 | +9.3 | 0.11 | **PASS** |
| trend | 5.48 | -27.3 | -25.2 | -50.8 | -17.3 | -28.6 | -0.73 | fail |
| fouling law | 4.26 | +1.0 | -9.9 | -4.4 | -2.5 | +19.2 | 0.09 | fail |
| XGBoost | 4.15 | +3.6 | -23.7 | +27.3 | -2.4 | +6.2 | 0.13 | fail |
| hybrid | 4.45 | -3.3 | -25.3 | +10.4 | -13.2 | +15.7 | -0.01 | fail |
| **Prophet** (new) | 4.46 | -3.6 | -49.4 | +20.9 | +1.0 | -17.8 | 0.15 | fail |
| **Prophet + level shift** (new) | 3.88 | +10.0 | -1.1 | +29.4 | -1.6 | +22.6 | 0.22 | fail |
| **LSTM** (new) | 5.04 | -16.9 | -46.6 | +9.2 | -1.7 | -64.1 | -0.06 | fail |
| **RNN** (new) | 4.82 | -11.8 | -39.5 | +8.9 | -4.0 | -37.6 | -0.03 | fail |

**ΔP, 1 d ahead, unseen train** (n = 642)

| method | MAE, pp | skill % | block 1 | block 2 | block 3 | block 4 | R² of the change | bar |
|---|---|---|---|---|---|---|---|---|
| no change | 1.76 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | -0.00 | — |
| smoothed no change | 2.13 | -20.8 | -27.7 | -52.0 | -14.1 | +1.7 | -0.35 | fail |
| trend | 1.84 | -4.5 | -6.6 | -5.2 | -7.1 | -1.1 | -0.09 | fail |
| fouling law | 2.62 | -48.5 | -68.1 | -50.0 | -62.2 | -27.3 | -1.01 | fail |
| XGBoost | 2.17 | -22.9 | -19.5 | -14.2 | -63.0 | -4.2 | -0.64 | fail |
| hybrid | 3.14 | -78.0 | -112.1 | -100.8 | -115.8 | -16.6 | -2.00 | fail |
| **Prophet** (new) | 7.23 | -309.4 | -690.5 | -159.4 | -297.1 | -219.0 | -13.57 | fail |
| **Prophet + level shift** (new) | 3.39 | -91.9 | -120.3 | -79.9 | -148.6 | -46.7 | -2.18 | fail |
| **LSTM** (new) | 1.82 | -3.0 | -3.8 | -5.6 | -2.2 | -1.3 | -0.06 | fail |
| **RNN** (new) | 1.85 | -4.7 | -13.7 | -5.0 | -1.9 | -1.3 | -0.09 | fail |

**ΔP, 3 d ahead, unseen train** (n = 617)

| method | MAE, pp | skill % | block 1 | block 2 | block 3 | block 4 | R² of the change | bar |
|---|---|---|---|---|---|---|---|---|
| no change | 2.83 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | -0.00 | — |
| smoothed no change | 2.76 | +2.2 | +8.7 | -5.6 | +0.9 | +8.9 | 0.04 | fail |
| trend | 3.18 | -12.5 | -14.2 | -12.3 | -19.3 | -4.4 | -0.26 | fail |
| fouling law | 3.00 | -6.3 | -13.3 | +12.5 | -15.9 | -17.1 | -0.10 | fail |
| XGBoost | 3.18 | -12.4 | -22.5 | +0.5 | -14.5 | -19.6 | -0.28 | fail |
| hybrid | 4.48 | -58.7 | -59.0 | +0.2 | -100.5 | -101.6 | -1.66 | fail |
| **Prophet** (new) | 7.51 | -165.7 | -346.7 | -27.7 | -171.3 | -190.4 | -5.66 | fail |
| **Prophet + level shift** (new) | 3.61 | -27.6 | -36.2 | +9.1 | -67.7 | -32.6 | -0.49 | fail |
| **LSTM** (new) | 3.10 | -9.7 | -15.1 | -4.5 | -9.3 | -12.7 | -0.17 | fail |
| **RNN** (new) | 3.24 | -14.7 | -29.9 | -1.3 | -28.6 | -5.7 | -0.38 | fail |

**ΔP, 7 d ahead, unseen train** (n = 572)

| method | MAE, pp | skill % | block 1 | block 2 | block 3 | block 4 | R² of the change | bar |
|---|---|---|---|---|---|---|---|---|
| no change | 2.91 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | -0.02 | — |
| smoothed no change | 2.86 | +1.8 | -3.8 | +5.0 | -2.8 | +6.0 | 0.07 | fail |
| trend | 3.68 | -26.4 | -45.2 | -29.6 | -31.7 | -4.7 | -0.61 | fail |
| fouling law | 3.39 | -16.2 | -76.5 | +9.7 | -10.9 | -25.9 | -0.17 | fail |
| XGBoost | 3.74 | -28.5 | -161.4 | +1.5 | -2.6 | -26.4 | -0.73 | fail |
| hybrid | 4.87 | -67.3 | -298.3 | +2.2 | -46.0 | -61.2 | -1.46 | fail |
| **Prophet** (new) | 7.91 | -171.6 | -583.9 | -29.7 | -150.6 | -166.6 | -5.31 | fail |
| **Prophet + level shift** (new) | 3.84 | -31.7 | -123.8 | +12.8 | -49.1 | -23.2 | -0.46 | fail |
| **LSTM** (new) | 3.85 | -32.0 | -215.5 | -0.6 | -2.2 | -7.4 | -1.03 | fail |
| **RNN** (new) | 3.24 | -11.4 | -74.0 | +0.7 | -1.2 | -5.0 | -0.14 | fail |

**ΔP, 14 d ahead, unseen train** (n = 502)

| method | MAE, pp | skill % | block 1 | block 2 | block 3 | block 4 | R² of the change | bar |
|---|---|---|---|---|---|---|---|---|
| no change | 3.49 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | -0.07 | — |
| smoothed no change | 3.36 | +3.6 | +9.9 | +1.5 | +2.4 | +4.0 | 0.02 | fail |
| trend | 5.34 | -53.2 | -72.5 | -79.3 | -42.1 | -18.3 | -1.52 | fail |
| fouling law | 4.20 | -20.4 | -91.8 | -6.7 | +5.8 | -34.2 | -0.43 | fail |
| XGBoost | 4.34 | -24.6 | -100.3 | -9.1 | +0.9 | -36.7 | -0.73 | fail |
| hybrid | 5.50 | -57.9 | -143.3 | -26.1 | -43.7 | -67.4 | -1.92 | fail |
| **Prophet** (new) | 8.75 | -151.0 | -404.4 | -57.1 | -134.8 | -131.7 | -5.43 | fail |
| **Prophet + level shift** (new) | 4.43 | -26.9 | -78.6 | -16.1 | -18.1 | -20.1 | -0.58 | fail |
| **LSTM** (new) | 4.27 | -22.6 | -100.5 | -18.4 | +15.4 | -37.9 | -0.51 | fail |
| **RNN** (new) | 4.30 | -23.4 | -110.8 | -18.6 | +15.0 | -33.3 | -0.53 | fail |

**flux loss, 1 d ahead, unseen train** (n = 642)

| method | MAE, pp | skill % | block 1 | block 2 | block 3 | block 4 | R² of the change | bar |
|---|---|---|---|---|---|---|---|---|
| no change | 2.11 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | -0.00 | — |
| smoothed no change | 2.44 | -15.9 | -24.9 | -42.8 | -8.0 | -8.7 | -0.20 | fail |
| trend | 2.17 | -3.1 | -3.8 | -4.5 | -2.6 | -2.8 | -0.05 | fail |
| fouling law | 3.05 | -44.9 | -34.6 | -79.0 | -45.0 | -29.7 | -0.32 | fail |
| XGBoost | 2.49 | -18.3 | -87.2 | -33.9 | -16.2 | -0.9 | 0.02 | fail |
| hybrid | 2.97 | -41.0 | -120.2 | -69.7 | -33.0 | -21.3 | -0.26 | fail |
| **Prophet** (new) | 4.03 | -91.4 | -274.7 | -68.5 | -58.6 | -103.6 | -0.80 | fail |
| **Prophet + level shift** (new) | 3.25 | -54.4 | -52.1 | -60.1 | -61.7 | -44.9 | -0.52 | fail |
| **LSTM** (new) | 2.24 | -6.4 | -42.1 | -8.3 | -2.0 | -3.7 | 0.06 | fail |
| **RNN** (new) | 2.24 | -6.4 | -47.8 | -7.9 | -2.6 | -2.5 | 0.06 | fail |

**flux loss, 3 d ahead, unseen train** (n = 617)

| method | MAE, pp | skill % | block 1 | block 2 | block 3 | block 4 | R² of the change | bar |
|---|---|---|---|---|---|---|---|---|
| no change | 3.07 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | -0.00 | — |
| smoothed no change | 3.02 | +1.4 | +10.0 | -7.4 | +8.3 | -0.1 | 0.05 | fail |
| trend | 3.34 | -8.9 | -4.5 | -12.0 | -9.0 | -7.5 | -0.16 | fail |
| fouling law | 3.48 | -13.6 | +3.3 | -15.5 | -26.6 | -3.6 | 0.00 | fail |
| XGBoost | 3.48 | -13.6 | -154.9 | +5.4 | -13.1 | -1.5 | 0.06 | fail |
| hybrid | 3.89 | -27.0 | -185.0 | -8.2 | -42.8 | +3.3 | -0.15 | fail |
| **Prophet** (new) | 4.09 | -33.4 | -144.6 | +11.7 | -35.7 | -42.8 | -0.12 | fail |
| **Prophet + level shift** (new) | 3.39 | -10.4 | +9.3 | +12.9 | -39.2 | -4.5 | 0.01 | fail |
| **LSTM** (new) | 3.42 | -11.6 | -48.8 | +1.9 | -25.7 | -1.3 | 0.02 | fail |
| **RNN** (new) | 3.35 | -9.1 | -67.3 | -2.9 | -9.3 | -2.6 | 0.06 | fail |

**flux loss, 7 d ahead, unseen train** (n = 572)

| method | MAE, pp | skill % | block 1 | block 2 | block 3 | block 4 | R² of the change | bar |
|---|---|---|---|---|---|---|---|---|
| no change | 3.48 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | -0.00 | — |
| smoothed no change | 3.39 | +2.7 | -13.3 | +8.0 | -5.3 | +7.7 | 0.04 | fail |
| trend | 4.25 | -22.1 | -27.0 | -17.5 | -26.8 | -21.0 | -0.40 | fail |
| fouling law | 3.75 | -7.9 | -28.7 | +1.9 | -35.8 | +13.4 | 0.06 | fail |
| XGBoost | 4.10 | -17.9 | -437.6 | +21.9 | -26.8 | +5.2 | 0.02 | fail |
| hybrid | 4.56 | -31.1 | -449.5 | +16.4 | -54.4 | -0.9 | -0.21 | fail |
| **Prophet** (new) | 4.24 | -21.8 | -250.6 | +22.9 | -39.7 | -17.2 | 0.03 | fail |
| **Prophet + level shift** (new) | 3.54 | -1.8 | -31.1 | +25.7 | -44.5 | +19.1 | 0.15 | fail |
| **LSTM** (new) | 4.31 | -23.9 | -180.7 | -2.0 | -41.4 | -7.4 | -0.08 | fail |
| **RNN** (new) | 4.34 | -24.7 | -184.5 | -12.6 | -32.2 | -9.1 | -0.03 | fail |

**flux loss, 14 d ahead, unseen train** (n = 502)

| method | MAE, pp | skill % | block 1 | block 2 | block 3 | block 4 | R² of the change | bar |
|---|---|---|---|---|---|---|---|---|
| no change | 4.31 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | -0.00 | — |
| smoothed no change | 4.00 | +7.1 | -0.1 | +16.4 | +3.5 | +9.3 | 0.11 | **PASS** |
| trend | 5.48 | -27.3 | -25.2 | -50.8 | -17.3 | -28.6 | -0.73 | fail |
| fouling law | 4.26 | +1.0 | -9.9 | -4.4 | -2.5 | +19.2 | 0.09 | fail |
| XGBoost | 4.24 | +1.7 | -24.1 | +21.0 | -2.2 | +3.0 | 0.12 | fail |
| hybrid | 4.58 | -6.3 | -45.1 | +19.7 | -12.4 | -0.5 | -0.04 | fail |
| **Prophet** (new) | 4.46 | -3.6 | -49.4 | +20.9 | +1.0 | -17.8 | 0.15 | fail |
| **Prophet + level shift** (new) | 3.88 | +10.0 | -1.1 | +29.4 | -1.6 | +22.6 | 0.22 | fail |
| **LSTM** (new) | 5.35 | -24.2 | -81.4 | -3.2 | -2.5 | -69.4 | -0.13 | fail |
| **RNN** (new) | 5.19 | -20.4 | -103.4 | -10.3 | -5.5 | -28.3 | -0.06 | fail |
