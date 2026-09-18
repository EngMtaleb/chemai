"""Project 1 - Distillation Soft Sensor.

Run:  python projects/p01_soft_sensor/train.py
"""
import json
import logging
from pathlib import Path

from chemai.config import SoftSensorConfig
from chemai.data import load_distillation_tower, add_periods, split_by_period
from chemai.data.loaders import check_derived_columns
from chemai.models import SoftSensor, save_model
from chemai.evaluation import evaluate, envelope_report, interval_coverage

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("p01")

OUT = Path(__file__).parent / "results.json"
MODEL_PATH = Path("models/soft_sensor.pkl")


def main() -> dict:
    cfg = SoftSensorConfig()

    df = load_distillation_tower(cfg=cfg)
    log.info("loaded %d rows", len(df))

    derived = check_derived_columns(df)
    if derived:
        log.warning("derived columns detected (excluded from features): %s", derived)

    df = add_periods(df, cfg)
    train, test = split_by_period(df, cfg)
    log.info("train %d rows (period %d) | test %d rows (period %d)",
             len(train), cfg.train_period, len(test), cfg.test_period)

    model = SoftSensor(cfg).fit(train)

    results = {
        "config": {
            "features": [cfg.temperature_col, cfg.pressure_col],
            "feature_form": "Antoine: 1000/T, plus operating pressure",
            "split": f"period {cfg.train_period} (train) vs {cfg.test_period} (test)",
        },
        "derived_columns_found": derived,
        "train": evaluate(model, train, "in-regime"),
        "test": evaluate(model, test, "out-of-regime"),
        "envelope": envelope_report(model, test),
        "uncertainty": interval_coverage(model, test),
    }

    # Ship the fitted model with the versions that produced it: a pickled
    # estimator is only valid for the scikit-learn that wrote it, and the
    # service reports a mismatch through /health.
    results["serialised"] = save_model(model, MODEL_PATH)

    OUT.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))

    print("\n--- inference examples ---")
    print("nominal :", model.predict_one(450.0, 228.0).to_dict())
    print("outside :", model.predict_one(520.0, 200.0).to_dict())
    return results


if __name__ == "__main__":
    main()
