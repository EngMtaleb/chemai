# Can the solar RO plant run off-grid?

*Off-grid study on the phase-2 model of the AI for RO deck (seawater RO, 1,000 m³/day, follows the sun, product-water tank). All numbers below are computed by the scripts in this folder from the phase-2 model and data, or copied from the cited sources.*

## Answer

**Yes, the model can run the plant on the sun alone, but not with the deck's design, and the water then costs more.** Off-grid here means zero grid kWh in every one of the ten years 2015–2024, with the demand always met.

- **The deck's design (2×, 0.8 kWp per m³/day, 12 h tank) cannot be off-grid.** Its sun share is 79.2 % (77.0–81.5 by year). No battery size closes the gap: over the ten years, the sun the plant cannot use, after the battery's 15 % loss, is less than the grid energy it would have to replace.
- **Two routes reach zero grid, at almost the same cost:**
  - **More water storage:** 2.45×, 2.15 kWp per m³/day, 77 h tank: **1.561 USD/m³**. A bigger plant fills a multi-day tank on sunny days.
  - **A battery:** 1×, 2.95 kWp per m³/day, 2 h tank + a 5,106 kWh / 449 kW battery: **1.563 USD/m³** at NREL's 2024 battery price (1.908 at the higher 2022 price behind NREL ATB 2024). A plant the size of the demand runs day and night; the battery carries the nights and the low-sun days.
  - The difference, 0.002 USD/m³, is well inside what the cost inputs can tell apart; which route is cheaper depends on what a bigger plant and a battery really cost (section 7).
  - Across the 8 cost cases of section 7, water storage is cheaper in 5 (base; only the RO system grows (30 %); discount rate 5 %; RO plant 1,200 USD per m3/d; battery 482 USD/kWh, 4-hour, O&M 2.5 % (NREL 2023 update, behind ATB 2024; 2022 USD)) and the battery in 3 (small steel tank, 750 USD/m3; RO plant 1,800 USD per m3/d; plant costs in 2024 USD (x 1.31, phase-2 notebook)).
- **Keeping the deck's 2× plant and adding a battery** costs more: 2×, 2.8 kWp per m³/day, 13 h tank + 3,364 kWh: 1.778 USD/m³.
- **The deck's design buying from the grid** costs 1.074, 1.110, 1.147, 1.183, 1.257 USD/m³ at grid prices of 0.05, 0.10, 0.15, 0.20, 0.30 USD/kWh. Going off-grid only pays if the grid costs more than **0.71 USD/kWh** (0.53–0.71 across the cost cases), above every grid price phase 2 considers (0.02–0.40 USD/kWh).

| Option | Design | Sun share | Panels' energy unused | Battery | USD per m³ |
|---|---|---|---|---|---|
| Deck design, grid makes up the rest | 2×, 0.8 kWp per m³/day, 12 h tank | 79.2 % | 20.0 % | – | 1.074 at 0.05 · 1.110 at 0.10 · 1.147 at 0.15 · 1.183 at 0.20 · 1.257 at 0.30 USD/kWh |
| Off-grid, water storage only (cheapest) | 2.45×, 2.15 kWp per m³/day, 77 h tank | 100.0 % | 61.9 % | – | 1.561 |
| Off-grid with a battery (cheapest found) | 1×, 2.95 kWp per m³/day, 2 h tank | 100.0 % | 70.0 % | 5,106 kWh, 449 kW | 1.563 |
| Off-grid with a battery, keeping the deck's 2× plant (cheapest found) | 2×, 2.8 kWp per m³/day, 13 h tank | 100.0 % | 70.4 % | 3,364 kWh, 232 kW | 1.778 |

Costs are the part of the water cost that the design decides (costs.py): RO plant capital and upkeep growing with capacity, panels, tank, battery, grid electricity. Chemicals, staff and other costs that every design pays alike are left out, as in phase 2. Battery sizes are per 1,000 m³/day of demand.

## 1. Method

The phase-2 scripts are imported read-only from the parent folder: `schedule.py` (demand 41.667 m³/h, 8 trains, energy and permeate curves per capacity, NASA POWER sun and sea temperature), `design_grid.py` (the fast rule and its scoring), `costs.py` (`annual_costs`, `per_m3`, `designs`, `SCENARIOS`) and `cost_inputs.json`. Nothing in the phase-2 folder was changed: the checksums of all its 77 files when this report was written equal those taken before the study (Python was stopped from writing `.pyc` files there).

**What I changed, and nothing else:**

1. **The ten years run as one series.** Phase 2 runs each year on its own, starting with an empty tank at midnight on 1 January. That start alone makes zero grid impossible: even a 4×, 6 kWp per m³/day, 168-hour design buys 981.5–986.6 kWh from the grid before the first sunrise of each year 2016–2024 (563.8 kWh in 2015, whose data start at 03:00). Here the tank is carried from one year to the next, and its level on 1 January 2015 is set equal to its level on 31 December 2024 (a repeating state, found by iteration). For the deck's design this moves the ten-year mean sun share from 79.19 % to 79.20 %.
2. **Part (b) only: a battery.** It charges only from sun the plant cannot use (phase 2's curtailed sun) and supplies the plant whenever the rule would buy from the grid. The rule still decides what the plant makes, exactly as in phase 2.

The physics (vessel model, curves, energy of the rest of the plant), the rule (by day make what the sun allows, up to capacity and tank space; at night or when the tank is empty make just the demand) and the cost method are unchanged.

**Smallest store.** For a given capacity and panel size, the water the sun alone can make each hour is fixed by the rule. The smallest tank that never runs dry is then found exactly with the sequent-peak method used to size reservoirs (the largest drawdown from a full tank, with the ten years repeating). The same calculation sizes the battery, with the battery's net flow (85 % of the spare sun it takes in, minus the grid energy it replaces). Both are checked against hour-by-hour simulation (section 2).

**Off-grid** means grid ≤ 1e-6 kWh in each year (the floating-point noise of the simulation) and no short hours.

**Costs.** `costs.annual_costs` with `cost_inputs.json` (discount rate 7.5 %, 25 years for plant, panels and tank), divided by the m³ made per year, as in phase 2. The battery is annualised the same way over its own life: capital × (CRF(7.5 %, 15 y) = 0.1133 + fixed O&M 4 %). No grid cost for the off-grid designs. Tanks are rounded up to whole hours and batteries to whole kWh and kW, and each rounded design is simulated hour by hour.

## 2. What I checked

1. **The phase-2 baseline, re-run with the phase-2 code:** deck design 79.19 % sun (76.98–81.48 by year); largest swept design (2×, 2.0 kWp, 24 h) 93.91 % (92.53–95.09), with 61.43 % of the panels' energy unused. Largest difference from `out/design_grid.csv`: 4.7e-10 (kWh or m³).
2. **My copy of the rule** (`model.rule_s0`, the same lines plus a start level) gives exactly `design_grid.rule_fast`'s output when it starts empty (largest difference 0 m³/h, four designs × ten years).
3. **Capacities above 2×.** `schedule.curves` keeps **8 trains** at every capacity and makes each one bigger (Q_tr = C × demand / 8: 10.42 m³/h per train at 2×, 20.83 at 4×). The energy and permeate per m³ at a given flux depend only on the flux and the sea temperature, so a capacity's curve is the 2× curve scaled by C/2. Checked for 2.5×, 3×, 4× and 5×: the curves scale exactly (largest deviation 2.3e-13), the flux window (f_min, f_max) is the same, the top output is C × demand × f_max(T), and the convexity check inside `curves` passes. The fast rule matches the slow reference rule `schedule.run_rule` hour by hour at 4× (4 kWp, 72 h, 2024) and 3× (2.5 kWp, 48 h, 2022): largest difference 0 m³/h. What this means: a 4× plant is modelled as 8 trains of 20.83 m³/h, not as 16 trains of today's size. That changes the turndown steps and the train starts, not the energy per m³; below 8 × f_min the model already lets part of a train run, as in phase 2.
4. **The exact smallest tank agrees with hour-by-hour simulation** in all 448 of 448 coarse designs: a design is off-grid in the simulation exactly when its tank is at least the exact minimum.
5. **The cheapest water-only design, simulated hour by hour:** zero grid in all ten years (largest year 3.3e-12 kWh), 0 short hours; the tank is full for 892–1131 hours each year, so no year leans on the repeating-state assumption. With the tank half an hour smaller (75.71 h) it buys 70.5 kWh in its worst year.
6. **The cheapest battery design, simulated hour by hour:** zero grid in all ten years, 0 short hours; the battery is full for 2511–2697 hours each year. With 99 % of its energy it buys 98.6 kWh over the ten years.

## 3. (a) Water storage only

**Smallest tank (hours of demand) for zero grid in every year**, exact; ∞ = impossible with any tank. Rows: capacity; columns: kWp of panels per m³/day.

| capacity | 1 | 1.5 | 2 | 2.5 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|---|
| 2× | ∞ | ∞ | ∞ | ∞ | 480.9 | 291.2 | 220.8 | 171.5 |
| 2.25× | ∞ | 487.3 | 241.3 | 135.3 | 81.1 | 36.3 | 26.3 | 23.4 |
| 2.5× | ∞ | 220.7 | 83.4 | 57.4 | 41.8 | 28.4 | 25.8 | 23.3 |
| 2.75× | 520.8 | 114.5 | 77.0 | 50.5 | 34.3 | 28.4 | 25.8 | 23.2 |
| 3× | 411.6 | 111.1 | 72.5 | 45.5 | 34.2 | 28.4 | 25.8 | 23.2 |
| 3.5× | 309.4 | 106.3 | 65.1 | 38.8 | 34.0 | 28.4 | 25.8 | 23.2 |
| 4× | 266.1 | 103.0 | 60.5 | 38.8 | 34.0 | 28.4 | 25.8 | 23.2 |

The coarse sweep simulated 448 designs hour by hour (capacity 2, 2.25, 2.5, 2.75, 3, 3.5, 4; panels 1, 1.5, 2, 2.5, 3, 4, 5, 6; tank 24, 36, 48, 72, 96, 120, 144, 168 h); 233 reach zero grid in all ten years (full results: `out/water_coarse.csv`). The fine search then took capacity 1.8–3.5 and panels 0.8–4 kWp per m³/day in steps of 0.05, each with its exact smallest tank (2275 pairs, 1796 of them possible; `out/water_fine.csv`).

**Cheapest:** 2.45×, 2.15 kWp per m³/day, 77 h tank (exact minimum 76.2 h, rounded up): **1.561 USD/m³** = RO plant 0.903 + RO upkeep 0.171 + panels 0.391 + tank 0.096. It uses 3.570 kWh/m³ and leaves 61.9 % of its panels' energy unused. The optimum lies inside the searched range, and it is flat: 26 designs are within 1 % of it, from 2.4× to 2.55×, 1.8–2.4 kWp, 66–127 h.

**Cheapest off-grid cost at each capacity** (best panels and exact tank): 1.95× 2.081 · 2× 1.937 · 2.2× 1.642 · 2.45× 1.560 · 2.7× 1.608 · 3× 1.713 · 3.5× 1.922 USD/m³. Below 1.95× nothing works in the searched range.

**What sets its tank:** the worst stretch of the ten years, 2020-11-25 16:00 to 2020-12-07 06:00 (11.6 days, Nov–Dec): on 12 of its 13 calendar days the sun could not make the day's water, 8 were low-sun days (phase 2's definition: under half the month's median), and over the stretch the sun alone made 72.7 % of the demand. The tank's lowest level in the ten years is 33 m³, in 2020.

### What stops the deck's 2× plant

- **A day only has so many sunny hours.** With unlimited panels the plant runs at full output whenever the sun is up. Even then a 1.5× plant makes only 87.2 % of the ten years' demand (monthly means 76.5–104.2 %), so it can never be off-grid. A 2× plant makes 116.2 %, but only 102.0 % in Jan and 103.1 % in Oct: almost no margin. At sea temperatures of 33 and 35 °C each train has a single usable point, its design flux (f_min = f_max = 1.0), so in the hot months the plant cannot run above its nominal capacity. In the cool months the trains can run up to 121.4 % of design flux, but the days are short.
- **With real panel fields the margin is gone for months.** At 2× the smallest workable field is 2.9 kWp per m³/day, and it needs a 563-hour tank. With 4 kWp the tank must carry 291 h of demand through a 181-day drawdown (2018-08-19 to 2019-02-16); with 6 kWp, 171 h over 165 days; with 8 kWp, 114 h. With 3 kWp the drawdown lasts 941 days, running across years (481 h). The cheapest 2× off-grid design costs 1.937 USD/m³.
- **It is the season, not a few cloudy days.** The largest phase-2 design (2×, 2.0 kWp, 24 h) buys from the grid on 209.2 days a year, up to 203 days in a row; its grid energy falls in Dec (19.5 %), Nov (16.6 %), Jan (15.5 %), Oct (15.4 %), Sep (13.9 %), Aug (8.8 %) and only 10.2 % in the other six months. Low-sun days are only 18.9 a year, at most 4 in a row (worst day 2023-11-15, 0.34 kWh per kWp). With more capacity a short spell sets the tank instead: for 2.45× with 2.15 kWp it is 11.6 days from 2020-11-25; for 2.5× and 3× with 2 kWp, 11.6 and 10.6 days from 2020-11-25 and 2020-11-26.

## 4. (b) A battery

**Inputs** (NREL, *Cost Projections for Utility-Scale Battery Storage: 2025 Update*, NREL/TP-6A40-93281, June 2025; 2024 USD): capital = 241 USD/kWh × energy + 372 USD/kW × power; fixed O&M 4% of capital a year; no variable O&M; round trip 85%; life 15 years. Exact quotes and pages are in section 9.

**Assumptions (mine):** the rated energy is the energy the battery delivers (NREL defines duration as output duration), so the 15 % loss is taken on charging; no self-discharge; one power rating for charging and discharging, chosen as the cheapest that still gives zero grid; fixed O&M as a share of this battery's own capital; the battery is replaced at the end of its life (annualised over 15 years). NREL's line was fitted to systems of 1–10 hours for the mainland United States; longer batteries are an extrapolation.

| Design | Grid it would buy, 10 years | Spare sun × 85 %, 10 years | Smallest battery (power not limiting) | Cheapest battery | Duration | Full cycles a year | USD per m³ |
|---|---|---|---|---|---|---|---|
| 2×, 0.8 kWp, 12 h | 2.677 GWh | 2.159 GWh | impossible | – | – | – | – |
| 2×, 1.2 kWp, 12 h | 1.634 GWh | 6.593 GWh | 17,956 kWh | 17,962 kWh, 352 kW | 51.0 h | 9.1 | 2.981 |
| 2×, 1.6 kWp, 24 h | 1.043 GWh | 11.446 GWh | 11,918 kWh | 11,926 kWh, 592 kW | 20.1 h | 8.7 | 2.496 |
| 2×, 2 kWp, 24 h | 0.790 GWh | 16.623 GWh | 6,533 kWh | 6,542 kWh, 833 kW | 7.9 h | 12.1 | 2.062 |

**The deck's design cannot be made off-grid with a battery:** over the ten years it would buy 2.677 GWh from the grid, but its spare sun, after the 15 % loss, gives only 2.159 GWh (80.7 % of the need). Year by year (spare sun after the loss, as a share of the grid energy to replace): 2015 76.3 %, 2016 100.2 %, 2017 83.8 %, 2018 74.1 %, 2019 86.1 %, 2020 94.7 %, 2021 91.1 %, 2022 64.3 %, 2023 66.1 %, 2024 75.9 %. It is short in 9 of the 10 years; only in 2016 does the year's total cover the need. Over the ten years the spare sun is short, so no battery size, however large, makes this design off-grid.

- 2×, 1.2 kWp, 12 h: the battery is sized by 2020-11-26 to 2020-12-07 (11 days, Nov–Dec).
- 2×, 1.6 kWp, 24 h: the battery is sized by 2020-11-27 to 2020-12-07 (10 days, Nov–Dec).
- 2×, 2 kWp, 24 h: the battery is sized by 2020-11-27 to 2020-12-07 (10 days, Nov–Dec).

**Search.** A screen of 728 designs (capacity 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5; panels 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.5, 3, 3.5, 4, 5, 6; tank 0, 3, 6, 12, 18, 24, 36, 48 h), of which 672 can be off-grid and 655 need a battery; then a local search from the best (23 designs), moving to any cheaper neighbour and halving the steps down to 0.05×, 0.05 kWp and 1 h (`out/battery_screen.csv`). The capacity of the cheapest design is at the lowest possible value, 1×: a plant smaller than the demand cannot meet it on hot days, when each train is held at its design point. 20 searched designs are within 1 % of the cheapest.

**Cheapest design with a battery: 1×, 2.95 kWp per m³/day, 2 h tank + 5,106 kWh / 449 kW** (11.4 h), **1.563 USD/m³** = RO plant 0.368 + upkeep 0.070 + panels 0.536 + tank 0.003 + battery 0.586. Smallest energy with power not limiting: 5,106 kWh. 119.9 full cycles a year; 108,025 kWh a year lost in the battery; 70.0 % of the panels' energy still unused. Its size is set by 2018-11-13 18:00 to 2018-11-16 06:00 (2.5 days). Simulated hour by hour: zero grid in all ten years, battery full 2511–2697 hours each year. At the price behind NREL ATB 2024 (482 USD/kWh for a 4-hour system, 2022 USD, O&M 2.5 %): 1.908 USD/m³ with 5,106 kWh.

**Cheapest with a battery that keeps the deck's 2× plant: 2×, 2.8 kWp per m³/day, 13 h tank + 3,364 kWh / 232 kW** (14.5 h), **1.778 USD/m³** = RO plant 0.737 + upkeep 0.140 + panels 0.509 + tank 0.016 + battery 0.376. Smallest energy with power not limiting: 3,363 kWh. 16.9 full cycles a year; 10,040 kWh a year lost in the battery; 70.4 % of the panels' energy still unused. Its size is set by 2018-11-14 05:00 to 2018-11-16 06:00 (2.1 days). Simulated hour by hour: zero grid in all ten years, battery full 7446–7969 hours each year. At the price behind NREL ATB 2024 (482 USD/kWh for a 4-hour system, 2022 USD, O&M 2.5 %): 2.015 USD/m³ with 3,364 kWh.

**Water or battery: the trade-off.** Storing one hour of demand costs 457 USD a year as water in the tank, but 5,288–5,652 USD a year as battery energy (the plant's energy for one hour of demand at night, 143–153 kWh, at 241 USD/kWh, before the power part). But the water has to be made while the sun shines, which needs a bigger plant: each extra 1× of capacity costs 160,116 USD a year (RO plant capital and upkeep, base inputs). The battery lets a plant the size of the demand run through the night instead. At the base inputs the two come out almost level; section 7 shows which way each input tips it.

## 5. (c) Not modelled: wind, solar thermal with heat storage

Other ways to bridge the nights and the low-sun season exist and are not modelled here, because the project has no model for them: wind turbines, and concentrating solar power with thermal (heat) storage. No numbers are given for them.

## 6. Compare (same cost method)

| | USD per m³ |
|---|---|
| Deck design (2×, 0.8 kWp, 12 h) buying from the grid at 0.05 USD/kWh | 1.074 |
| Deck design (2×, 0.8 kWp, 12 h) buying from the grid at 0.10 USD/kWh | 1.110 |
| Deck design (2×, 0.8 kWp, 12 h) buying from the grid at 0.15 USD/kWh | 1.147 |
| Deck design (2×, 0.8 kWp, 12 h) buying from the grid at 0.20 USD/kWh | 1.183 |
| Deck design (2×, 0.8 kWp, 12 h) buying from the grid at 0.30 USD/kWh | 1.257 |
| Off-grid, water only: 2.45×, 2.15 kWp per m³/day, 77 h tank | 1.561 |
| Off-grid with a battery: 1×, 2.95 kWp per m³/day, 2 h tank + 5,106 kWh | 1.563 |
| Off-grid with a battery, deck's 2× plant kept: 2×, 2.8 kWp per m³/day, 13 h tank + 3,364 kWh | 1.778 |

Without any grid cost the deck's design costs 1.037 USD/m³ and buys 0.733 kWh/m³ from the grid (phase 2's year-by-year runs, `out/design_grid.csv`). Phase 2's grid prices are 0.02–0.40 USD/kWh (`costs.py`); all twenty are in `results.json`. The off-grid designs cost as much as the deck's design would at **0.71 USD/kWh** (water only), 0.72 USD/kWh (battery) and 1.01 USD/kWh (battery, 2× plant kept). Off-grid water costs 0.451–0.668 USD/m³ more than the deck's design buying from the grid at 0.10 USD/kWh.

## 7. Sensitivities (each option re-optimised)

Phase 2's cost scenarios (`costs.SCENARIOS`, except "spare trains already there", which assumes the extra trains already exist and so does not apply to building a bigger plant), phase 2's notebook case with the plant costs in 2024 USD (× 1.31), and the older NREL battery price. The water-only optimum is re-picked from the fine search (tanks at their exact minimum, so the base row shows 1.560 against 1.561 for the 77-hour tank); the battery optimum from the screened and searched designs. USD per m³; the last two columns are the grid prices (USD/kWh) at which the deck's design would cost as much as each off-grid route.

| Case | Deck + grid at 0.10 | Water only (design) | Battery (design) | Break-even grid price, water only | Break-even grid price, battery |
|---|---|---|---|---|---|
| base | 1.110 | 1.560 (2.45×, 2.15 kWp, 76 h) | 1.563 (1×, 2.95 kWp, 2 h, 5,106 kWh) | 0.71 | 0.72 |
| only the RO system grows (30 %) | 0.852 | 1.166 (2.65×, 1.65 kWp, 104 h) | 1.363 (2.5×, 2.5 kWp, 48 h, 1,166 kWh)* | 0.53 | 0.80 |
| small steel tank, 750 USD/m3 | 1.208 | 2.013 (2.4×, 3.5 kWp, 35 h) | 1.579 (1×, 2.95 kWp, 2 h, 5,106 kWh) | 1.20 | 0.61 |
| discount rate 5 % | 0.928 | 1.285 (2.45×, 2.15 kWp, 76 h) | 1.325 (1×, 3 kWp, 2 h, 5,046 kWh) | 0.59 | 0.64 |
| RO plant 1,200 USD per m3/d | 0.963 | 1.378 (2.5×, 1.95 kWp, 90 h) | 1.489 (1×, 2.95 kWp, 2 h, 5,106 kWh) | 0.67 | 0.82 |
| RO plant 1,800 USD per m3/d | 1.258 | 1.740 (2.45×, 2.15 kWp, 76 h) | 1.637 (1×, 2.95 kWp, 2 h, 5,106 kWh) | 0.76 | 0.62 |
| plant costs in 2024 USD (x 1.31, phase-2 notebook) | 1.382 | 1.892 (2.4×, 2.35 kWp, 69 h) | 1.699 (1×, 2.95 kWp, 2 h, 5,106 kWh) | 0.80 | 0.53 |
| battery 482 USD/kWh, 4-hour, O&M 2.5 % (NREL 2023 update, behind ATB 2024; 2022 USD) | 1.110 | 1.561 (same as base) | 1.775 (2.25×, 3.5 kWp, 36 h, 590 kWh) | 0.71 | 1.01 |

\* the re-picked design is at the edge of the searched range (water only: capacity 1.8–3.5×, panels 0.8–4.0 kWp; battery: the screened designs), so the true optimum of that case may lie outside it and cost a little less.

## 8. Limits

- **Ten years of weather, no margin.** The sizes are the smallest that survive the worst stretch of 2015–2024. A worse stretch outside these years would need more storage; no safety margin is added, and the plant is assumed available 100 % of the time (no outage, no membrane cleaning stop). A real off-grid plant would add a margin or a backup.
- **The phase-2 limits carry over:** no dust on the panels and no panel ageing; no membrane fouling or ageing; the rest of the plant as a fixed amount per m³; NaCl standing in for seawater; NASA POWER's sea skin temperature on a 2 °C table; the wear of train starts not counted.
- **The rule is kept.** For water storage alone the rule is already the best possible: it makes all the water the sun allows whenever there is space, so no other sun-only schedule can keep the tank fuller. With a battery, a schedule planned together with the battery might need a smaller battery; not tested.
- **Capacity above 2× is modelled as 8 bigger trains** (section 2). The energy per m³ is unaffected; as in phase 2, part of a train may run (at dawn and dusk the plant makes any small output at the f_min energy per m³). The cost assumes the whole plant grows with capacity (base case of `cost_inputs.json`); the "only the RO system grows" case is in section 7.
- **Price years differ:** RO plant and upkeep in 2016 USD (World Bank 2019), panels and battery in 2024 USD; the × 1.31 case in section 7 puts the plant in 2024 USD.
- **Battery:** NREL's costs are for the mainland United States and were fitted to 1–10-hour systems; the fixed O&M is meant to keep the rated capacity at about one cycle a day; no self-discharge; the round-trip loss is all on charging.
- **This study uses the phase-2 scripts** (`schedule.py`, `design_grid.py`), as asked. The phase-2 notebook (`make_notebook.py`) is a later variant (35 % recovery on the hottest days, among others); its text quotes 79.6 % sun for the deck's design (77.6–81.9 % by year), while the scripts give the 79.2 % used here. If the deck quotes the notebook's numbers, the off-grid sizes here come from the slightly different script model.

## 9. Sources

- **Battery, base:** W. Cole, V. Ramasamy, M. Turan, *Cost Projections for Utility-Scale Battery Storage: 2025 Update*, NREL/TP-6A40-93281, June 2025, <https://docs.nrel.gov/docs/fy25osti/93281.pdf>. p. 3: "All cost values were converted to 2024$ using the consumer pricing index." and "We define the 4-hour duration as the output duration of the battery, such that a 4-hour device would be able to discharge at its rated power capacity for 4-hours." p. 6: "...we used the bottom-up cost model to calculate the cost of a storage system with durations ranging from one hour to ten hours..."; "In the case of Figure 2, the energy cost is $241/kWh, and the power cost is $379/kW adjusted down by $7/kW for a final power cost of $372/kW." p. 8: "The 2024 starting point of $334/kWh is derived from the bottom-up cost model described in Section 2.2." (4-hour system; Table 2, p. 16: 2024, 334 USD/kWh). p. 11: "The FOM value selected is 4% of the $/kW capacity cost for a 4-hour battery."; "...we have adopted zero for the VOM." p. 12: "The round-trip efficiency is chosen to be 85%, which is well aligned with published values."; "The lifetime we selected is 15 years, which is on the lower end of published values."
- **Battery, sensitivity (the cost projections that NREL ATB 2024 cites):** W. Cole, A. Karmakar, *Cost Projections for Utility-Scale Battery Storage: 2023 Update*, NREL/TP-6A40-85332, June 2023, <https://www.nrel.gov/docs/fy23osti/85332.pdf>. p. 2: "All cost values were converted to 2022$ using the consumer pricing index." p. 4: "The 2022 starting point of $482/kWh is taken from Ramasamy et al. (2022)." (Table 2, p. 13: 2022, 482 USD/kWh, 4-hour). p. 7: "The FOM value selected is 2.5% of the $/kW capacity cost for a 4-hour battery." p. 8: round trip 85 %, lifetime 15 years. NREL ATB 2024, utility-scale battery storage, <https://atb.nrel.gov/electricity/2024/utility-scale_battery_storage>: 85 % round trip, 15-year lifetime, FOM 2.5 % of the capital cost in $/kW, cost breakdown in 2022$; it cites Cole and Karmakar (2023).
- **Sun and sea temperature:** NASA POWER hourly and daily point data 2015–2024 (phase-2 `data/`).
- **All other costs:** phase-2 `cost_inputs.json` and its `_sources` (World Bank 2019, *The Role of Desalination in an Increasingly Water-Scarce World*; IRENA, *Renewable Power Generation Costs in 2024*; Kocher & Menon 2023). The × 1.31 factor (plant costs from 2016 to 2024 USD) is phase 2's notebook case (US consumer price index, Bureau of Labor Statistics).

## 10. Files

- `model.py`: shared helpers (imports the phase-2 code; start level; repeating state; sequent peak; battery)
- `check_model.py` → `out/check_model.json`: the checks in section 2
- `sweep_water.py` → `out/water_coarse.csv`, `out/water_fine.csv`, `out/water_result.json` (log `out/water.log`): part (a)
- `battery.py` → `out/battery_designs.csv`, `out/battery_screen.csv`, `out/battery_result.json` (log `out/battery.log`): part (b)
- `compare.py` → `results.json`, `out/compare.json`, `out/compare_table.csv`: the comparison and sensitivities
- `make_report.py` → this file

