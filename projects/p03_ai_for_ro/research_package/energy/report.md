# Industrial RO solar and storage study — executed 30 October 2026

## Evidence and objective

This is a newly executed engineering scenario study. Water and electricity relations come from the industrial physics model; the solar source is **PVGIS-ERA5 modeled PV**, not a photovoltaic meter. The daylight rule and MILP controller are not learned AI. No real plant was controlled. Four predetermined 72-hour windows do not establish annual savings.

## Common plant and fair comparison

Three switchable trains have 600 m³/day gross nominal capacity; the main demand is 400 m³/day. All policies have the same 200 m³ water tank, 320 kWp PV, no battery, no export and unrestricted grid backup. Initial and required final stock are both 100 m³, with a 20 m³ reserve. The third train provides explicit installed headroom. Capital cost and long-term duty rotation are outside scope.

ON trains modulate only within the physics-checked 6.888–8.333 m³/h interval. The bounded affine power relation multiplies its intercept by the ON state and is never extrapolated to zero flow. Its endpoints require 19.21365–24.01093 kW/train and pass the physics operating/product-salt gates. At the midpoint, affine power overestimates the physical calculation by 0.1761%. The 500 mg/L salt screen is an illustrative process specification, not potable-water certification. Feed remains 25°C and 43 g/L; outdoor temperature is not substituted for feed temperature.

A routine start diverts the first 5 minutes of modeled product while full production-associated electricity is charged. This and a 2-hour minimum ON time are study assumptions, not measured or OEM-validated transitions. No chemical CIP or post-CIP rinse is simulated. OFF production-associated power is zero. Standby loads, water aging, intake/pretreatment yield losses, chemicals and dynamic flushing hydraulics are not modeled. Simplified feed-pump electricity and a 0.45 kWh/m³ auxiliary allowance come from the physics study.

## Policies

1. **Fixed schedule:** two fully loaded trains continuously supply 400 m³/day. The 600 m³/day stress uses three, despite the retained internal identifier `fixed_two_trains`.
2. **Daylight + tank rule:** nominal clock plan operates one train from 23:00–06:00, three from 08:00–17:00, and two otherwise. Its flow trim balances known demand and routine-start losses. Inventory feedback corrects departures from this reference over up to 6 hours, respecting finite storage and minimum ON time. It sees no future weather.
3. **Rolling optimization:** a MILP replans hourly over up to 24 hours. It minimizes forecast grid use, with small total-electricity and startup tie-break terms. PV forecasts repeat the previous observed 24 hours. A median of the last four completed nonzero-output measurements estimates accepted-yield loss. Tank measurements are exact in this simulation. Every horizon ends at the common target stock. A failed solve uses the logged rule fallback.

MILP settings: 0.5% requested relative gap and 3-second time limit per call. Twelve calls returned time-limited feasible incumbents in this run; each was independently checked against the full constraints. These are not proven global optima. Retained trajectories reproduce the stated accounting exactly; a rerun can choose a slightly different feasible schedule depending on solver timing/hardware.

The checks separate unserved water, reserve intrusion, spillage and terminal stock error. A schedule failing any water/stock condition is excluded from energy-saving rankings. Ending with depleted stock cannot count as a saving.

## Seasonal results

|Window|Policy|Grid kWh|Total kWh|PV share|Stock/service pass|
|---|---|---:|---:|---:|---|
|2023-01|Fixed schedule|2332.03|3457.57|32.55%|True|
|2023-01|Daylight + tank rule|1988.13|3449.00|42.36%|True|
|2023-01|Rolling optimization|1910.84|3405.07|43.88%|True|
|2023-04|Fixed schedule|1933.43|3457.57|44.08%|True|
|2023-04|Daylight + tank rule|1400.87|3449.00|59.38%|True|
|2023-04|Rolling optimization|1252.11|3417.92|63.37%|True|
|2023-07|Fixed schedule|1888.23|3457.57|45.39%|True|
|2023-07|Daylight + tank rule|1351.98|3449.00|60.80%|True|
|2023-07|Rolling optimization|1174.40|3419.67|65.66%|True|
|2023-10|Fixed schedule|2002.39|3457.57|42.09%|True|
|2023-10|Daylight + tank rule|1496.93|3449.00|56.60%|True|
|2023-10|Rolling optimization|1317.64|3419.67|61.47%|True|

For October, rolling optimization reduces grid import by **34.20%** versus fixed operation; total modeled electricity changes by **-1.10%**. These are different denominators and different claims. Most of the gain comes from shifting production toward PV with installed headroom and storage. The smaller total-energy change comes from the bounded operating point and startup accounting. It is not an AI-forecast achievement or measured industrial saving.

## Disturbances and limits

|Case|Policy|Unserved m³|Terminal error m³|Grid kWh|Pass|
|---|---|---:|---:|---:|---|
|october_yield90|Fixed schedule|0.000|-80.000|2002.39|False|
|october_yield90|Daylight + tank rule|0.000|-2.193|1671.30|False|
|october_yield90|Rolling optimization|0.000|-0.000|1564.24|True|
|october_forecast140|Fixed schedule|0.000|0.000|2002.39|True|
|october_forecast140|Daylight + tank rule|0.000|-0.000|1496.93|True|
|october_forecast140|Rolling optimization|0.000|0.000|1316.19|True|
|october_capacity_limit_yield90|Fixed schedule|0.000|-120.000|3082.88|False|
|october_capacity_limit_yield90|Daylight + tank rule|0.000|-120.000|3082.88|False|
|october_capacity_limit_yield90|Rolling optimization|0.000|-120.000|3082.88|False|

The accepted-yield step to 0.9 at hour 24 deliberately removes useful product while retaining the same electrical draw. Extra unavailable/diverted product is accounted separately from routine-start diversion. This is not a fitted fouling law, a new membrane solution at changed recovery, or a rate estimated from the industrial observations.

At 400 m³/day demand, the rolling controller restores the required final stock after this stress. The fixed and simple-rule policies leave terminal deficits, so their energy numbers are not like-for-like savings. At 600 m³/day demand, a 600 m³/day plant has no capacity to recover sustained yield loss. Initial storage hides the shortfall temporarily: all policies serve demand during this 72-hour window, but all end 120 m³ below target. That is not sustainable steady operation.

The +40% PV forecast stress can improve or worsen a particular realized schedule by chance; it does not establish a general benefit from biased forecasts. Other October sensitivities use 6/12/24 hours of storage and 240/320/400 kWp PV. The extra tank has little benefit in this particular replay once storage is sufficient. No tariff, capital-cost optimum or off-grid autonomy is claimed.

## Verification

The separate physics agent independently reconstructed all 33 trajectories, confirming gross water, startup loss, accepted product, tank, PV, grid and load accounting, as well as every routine start's minimum ON time. Independent pass/fail classifications match: 28 pass, 5 fail. Across 25 physical points within the trim interval, every operating gate passes and the largest power interpolation error is 0.17633%. Review code and output are in `physics/review_energy.py` and `physics/energy_independent_review.json`.

## Reproduction and source

Run `python energy/run_study.py`, then `python energy/make_report.py` from the bundled project. Dependencies: NumPy, Pandas, SciPy and Matplotlib. The sibling `physics/model.py` contains the engineering model. All 8,760 PV hours, original API JSON and metadata are retained in `data/`; no network is required. Every trajectory and solver status is in `results/`. `plot_data.json` contains display arrays; `manifest.json` records file hashes.

Source acknowledgment: **Photovoltaic Geographic Information System (PVGIS) of the European Commission Joint Research Centre**. Public Kuwait City reference point 29.3759°N, 47.9774°E; year 2023; fixed 30° south-facing free-standing crystalline silicon; 14% system-loss setting. Original 4.5 kWp modeled output is scaled linearly to each studied array size. ERA5 timestamps are hourly centers at HH:30 UTC, displayed as hourly intervals in Kuwait (UTC+3). `sources.json` retains the request, hashes and interpretation.

- https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis_en
- https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/using-pvgis-5/pvgis-5-user-manual_en
- https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/general-information/usage-conditions-data-protection_en

**Arabic narration:** «ثبتنا نفس المحطة ونفس الطلب ونفس حجم الخزان عند المقارنة. شغّلناها بثلاث طرق: تشغيل ثابت، وقاعدة بسيطة مرتبطة بالنهار والخزان، وتحسين يعيد حساب الخطة كل ساعة. استفدنا أكثر من الشمس، لكن فرقنا بين تقليل كهرباء الشبكة وتقليل استهلاك التحلية نفسه. وحين نقص الماء المقبول، اختبرنا هل النظام يقدر يعوضه ويرجع الخزان لنفس المستوى. النتائج محاكاة صناعية بشروط واضحة، وتوضح دور التخطيط والمرونة؛ ما ننسب كل فائدة للذكاء الاصطناعي.»
