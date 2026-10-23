# Project 4 — Does the tide foul the membrane?

*Draft — the headline and results are written after the declared test is run.*

## The question

Seawater desalination plants draw from the sea, and the sea follows the moon.
If the fouling load at an open intake rises and falls with the tide, part of
the plant's future fouling is known years in advance — something no sensor and
no forecast model can give. Backwash, coagulant dose and production could then
be planned against the tide instead of chasing it.

The test is built so that it cannot be fooled by the day. A process can swing
every ~12 hours for many reasons — shifts, sunlight, temperature. Only the tide
keeps the same phase against the astronomical clock across runs months and
years apart. The test asks for exactly that, and uses no tide gauge and no
location: only the periods of the tidal constituents, which are astronomy.

## Data

Public UF pilot record, Zenodo 4630769 (CC0), also Dryad
[doi:10.5068/D1310B](https://doi.org/10.5068/D1310B): an ultrafiltration unit
pre-treating seawater for a reverse-osmosis pilot, 2012–2015. The site is not
named here. After the audit (VALIDATION.md 1): 107 runs with a trusted clock,
5,106 h, cut into 14,998 filtration segments.

## Method — declared before the test was run

See [DECLARED_BEFORE_RUN.md](DECLARED_BEFORE_RUN.md), committed before any
tide term touched the real data.

1. **Fouling rate per segment** — slope of ln R, R = TMP / (μ(T)·Q).
2. **Base model** — run levels, slow trend per episode, time of day
   (24/12/8/6 h), temperature, flow, segment length, element pair.
3. **Tide block** — M2, N2, O1, M4, amplitude and phase fitted.
4. **Null** — the same fit with each block of episodes moved by whole weeks:
   time of day and day of week stay, the moon is scrambled. 999 surrogates.
5. **Checks** — positive controls (pump speed, pump discharge pressure),
   placebo periods (11.0, 13.5, 27.0, 5.3 h), out-of-time confirmation
   (fit 2012–2013, freeze, score 2014–2015), power analysis.

## Run it

```bash
pip install -e ".[viz]"
python -c "from chemai.data.uf_pilot import download_archive, extract_archive; extract_archive(download_archive())"
python projects/p04_tide_fouling/step1_audit.py
python projects/p04_tide_fouling/step2_segments.py
python projects/p04_tide_fouling/step3_power.py
python projects/p04_tide_fouling/step4_tide_test.py
```

`results/segments.parquet` is committed (1.9 MB), so steps 3–4 run without
the 356 MB archive.
