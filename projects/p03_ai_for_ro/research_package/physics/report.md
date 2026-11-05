# Industrial RO physics study — recomputed 30 October 2026

## What this adds

A reproducible industrial-sized **scenario model**, replacing the assumption that a single laboratory element can simply stand in for a plant. The model is not a commissioned plant design or a validated digital twin. Actual industrial fouling observations are analyzed separately. The stress multipliers below are deliberately chosen engineering scenarios; they are not estimated from that industrial dataset.

**Arrangement:** three independently switchable trains; two parallel pressure vessels per train; seven SW30HRLE-440i 8-inch elements per vessel; 41 m² per element. Total area is 1,722 m². Nominal gross RO production is 600 m³/day at 25°C, 43 g/L feed salinity, 40% recovery and 14.518 LMH mean flux. Each train delivers 8.333 m³/h. This is a study design, not the arrangement of the real-data source plant.

The system boundary starts at the low-pressure RO feed and ends with blended RO permeate and depressurized brine. Intake, pretreatment, chemical dosing and product post-treatment appear in the process story; their detailed hydraulic yields and chemical performance are not simulated. A 3-bar feed-pump term and an explicitly assumed 0.45 kWh per gross m³ ancillary-electricity term account for simplified upstream/downstream electric loads. The separate energy-dispatch study counts startup diversion and accepted production. Gross RO flow here does not itself establish delivered drinking-water quality.

## Equations and calibration

The element is split into 16 axial area slices. For each slice:

- Water transport: `J = A(T) [P − Pp − π(Cm,T) + π(Cp,T)]`.
- Salt transport: `J Cp = B(T) (Cm − Cp)`.
- Film concentration polarization: `Cm = Cp + (Cb − Cp) exp(J/k)`.
- Permeate volume: `dQp = J dArea /1000` in m³/h for J in LMH.
- Salt and volume are conserved before entering the next slice.
- Channel pressure loss is an assumed flow/viscosity correlation, not a measured vessel pressure-loss curve.

The manufacturer's reference is one 41 m² element delivering 30.2 m³/day with 99.8% stabilized rejection at 55 bar, 32 g/L NaCl, 25°C and 8% recovery. Fitting A and B to this one point gives **A25 = 1.182518 LMH/bar** and **B25 = 0.051761 LMH**. The benchmark is recovered to numerical precision, because it is the calibration target. It is **not independent evidence of industrial accuracy**.

Temperature correction for water permeability and the approximate total-salt osmotic-pressure relation come from the DuPont technical manual. Salt-permeability temperature response, mass transfer, hydraulic pressure loss and efficiencies are disclosed study assumptions. Sensitivity tests expose these choices rather than interpreting them as measured properties. No ion-resolved scaling, boron, microbiology or chemical pretreatment model is included.

## Recomputed nominal point

| Quantity | Result |
|---|---:|
| Plant gross permeate | 600.000 m³/day |
| Feed pressure | 63.490 bar |
| Pressure loss per vessel | 1.014 bar |
| Blended permeate salt | 227.580 mg/L |
| Concentrate salt | 71.515 g/L |
| RO high-pressure + booster electricity | 2.1535 kWh/m³ |
| Total modeled electricity incl. feed + assumed auxiliaries | 2.8813 kWh/m³ |
| Plant power | 72.0328 kW |
| Feed / brine per train | 20.833 / 12.500 m³/h |

Per train, high-pressure pump power is 16.4734 kW, ERD booster 1.4727 kW, feed pump 2.3148 kW and assumed auxiliaries 3.7500 kW. ERD pressure-transfer efficiency is assumed to be 96%; HP pump 85%; booster 80%; feed pump 75%. Recovery of brine pressure is a hardware benefit, not an AI benefit.

Inside a vessel, the seven element-average fluxes are approximately 26.27, 21.51, 17.15, 13.33, 10.14, 7.59 and 5.63 LMH. Salt concentration rises along the brine channel as permeate leaves. The blended permeate gate applies to the volume-weighted vessel/plant output, not each separate element outlet.

## Stress results at the same gross water production

| Scenario assumption | Pressure bar | SEC kWh/m³ | Blended salt mg/L | Study gates |
|---|---:|---:|---:|---|
| Clean, 25°C, 43 g/L |63.490|2.8813|227.58|Pass|
| Winter,15°C |67.233|3.0274|152.10|Pass|
| Hot,35°C |61.933|2.8162|340.49|Pass|
| Feed50g/L |72.444|3.1919|266.40|Pass|
| Water permeability A×0.8 |66.052|2.9713|225.35|Pass|
| Channel resistance law×3 |64.873|3.0295|228.58|Pass|
| Salt permeability B×2.5 |62.958|2.8630|562.59|Fail salt gate|
| A×0.7,B×1.5,channel×2 |68.498|3.1088|335.65|Pass|
|35°C and B×2.5 |61.007|2.7843|835.24|Fail salt gate|
|20LMH instead of14.518 |68.580|3.0971|165.26|Fail feed-flow/element-flux gates|
|HP pump75%efficiency |63.490|3.1449|227.58|Pass|
|No energy recovery |63.490|5.6698|227.58|Pass|

An instructive result is that higher salt passage can slightly lower the calculated energy, while failing the product-salt gate. An energy optimizer must not accept that state simply because its kWh/m³ is lower. Likewise, temperature may reduce pressure demand while increasing salt passage at fixed production. These are simulated conditional relationships, not observed plant savings.

## Operating envelope and checks

The offline grid contains **540** points: 3 assumed membrane states ×5 temperatures(15–35°C) ×4 feed salinities(38,43,48,53g/L) ×3 recoveries(35,40,45%) ×3 fluxes(12,13.26,14.518LMH). **389/540** pass the stated model gates. This fraction describes the chosen grid, not a plant reliability probability.

Manufacturer maximum pressure drops are enforced:1bar/element,3.5bar/vessel. The study conservatively limits feed pressure to80bar and temperature to35°C. Additional design gates are explicitly study assumptions:32LMH element-average flux,13%element recovery,13m³/h maximum vessel feed,3.5m³/h minimum vessel brine and500mg/L blended product salt. The salt gate is an illustrative process specification, not a potable-water certificate. DuPont's special iLEC installation requirements above35°C/42bar are outside this study temperature envelope.

Numerical verification across scenario/grid results found maximum absolute volume residual7.46e−14m³/h and salt residual7.76e−12kg/h. Refining16 to64slices/element changes nominal pressure0.123%, product salt0.153% and SEC0.090%. These small errors concern numerical consistency only; parameter uncertainty and unmodeled physics are larger limitations.

Additional review checked the numerical safety caps in all540published grid solutions: maximum final `J/k` was0.22775 against an exponential cap at5; maximum raw segment permeate/feed fraction was0.013015 against a flow cap at0.25. Thus neither cap changes any published grid result. An independent conservative bound across540grid+13scenario+24sensitivity points also remains below both caps. This verifies implementation behavior, not physical realism of the assumed correlations.

The low-pressure numerical branch allows extremely small forward water flux with nearly feed-like permeate salinity because finite salt permeability reduces the osmotic difference. It does not represent freshwater production below the osmotic requirement. No reverse-flow, startup transient or membrane salt-accumulation dynamics are simulated; the published operating domain is well away from this branch and includes the blended-product salt gate.

## Dispatch interface

`from model import solve_point, DEFAULT`

`solve_point({'target_flux_LMH':12, 'temperature_C':25, 'feed_salinity_g_L':43})`

Returns per-train production, power, pressure, SEC, product salt, constraint checks and axial profiles. The energy agent uses the endpoints of the feasible12–14.518LMH interval for ON-train modulation. At25°C43g/L:

- Low:6.888m³/h,19.21365kW.
- High:8.33333m³/h,24.01093kW.
- Exact midpoint21.57430kW versus linearly interpolated21.61229kW, an overestimate of0.1761% at the midpoint.

The interpolation is an optimization surrogate over that bounded interval. Its negative intercept must never be extrapolated to an OFF train; OFF is zero production and zero production-associated power in that model. The dispatch study separately discloses standby/startup assumptions.

## Reproduction and source files

Run `python run_study.py` in this directory (Python with NumPy, SciPy and Matplotlib). No downloads are needed. `model.py` contains all equations and assumptions; `results.json` contains detailed numerical results; `operating_grid.json` contains precomputed scenario selections; `dispatch_endpoints.json` is the energy interface. `sources.json` records primary URLs, reference conditions and the boundary between sourced and assumed parameters. SVG/PNG figures were rendered and visually checked.

**Recommended narration:** "كبّرنا السؤال إلى محطة صناعية افتراضية بمعدات وأبعاد واضحة. حسبنا تغير الأداء داخل الأوعية، وراجعنا موازين الماء والملح وحدود التشغيل. النتائج هنا محاكاة مرتبطة بمواصفات منشورة، ومعايرتها على نقطة الشركة لا تجعلها محطة حقيقية أو دليل توفير ميداني. بيانات المصنع الفعلية نستخدمها في اختبار منفصل للتلوث، وكل نوع من الأدلة ظاهر باسمه."
