# Reproduce the industrial-derived indicator study

This folder is a self-contained, public-data analysis. It contains the verified Zenodo deposit, new analysis code, a frozen study plan, metrics and plots.

1. Use Python3.12 with numpy2.3.5, pandas2.2.3, scipy1.17.0, scikit-learn1.8.0 and matplotlib3.10.8.
2. Run `python run_study.py`. No network or API key is required.
3. Review `results.json`, `metrics.csv`, `metrics_by_split.csv` and `report.md`. Full per-row pairs and predictions are regenerated as CSV.

The independently downloaded industrial daily data are not raw SCADA. Exact source semantics and limitations are documented in report.md. DATA5.json is the JSON payload extracted from the user-provided reference presentation, used only to compare rounded embedded observations against the independently downloaded source. It is not used to train the forecasting models.

The source dataset is CC-BY-4.0. Source deposit code is MIT. All source license and attribution files are retained. This analysis does not prescribe autonomous CIP or validate an industrial energy saving.
