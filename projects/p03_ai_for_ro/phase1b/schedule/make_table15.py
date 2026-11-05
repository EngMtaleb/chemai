"""declaration_v15.md, the table: make_table.py unchanged except the calibrated measurement (calibration15.json):
screen flow exponent 1.75, transmitter noise 0.5x-1.5x, outliers p, slow movement w. Writes poc15/data/."""
import dataclasses, hashlib, json, sys
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
SP = HERE.parent
sys.path.insert(0, str(SP / "phase3b")); sys.path.insert(0, str(SP / "phase3b" / "poc"))
import sim_v2 as S, make_table as MT
from chemai.config import ROConfig, ROSimConfig

CAL = json.loads((HERE / "out" / "calibration15.json").read_text())["chosen"]
RO = dataclasses.replace(ROConfig(), vendor_flow_exponent=1.75)
CFG = dataclasses.replace(ROSimConfig(days=S.N_DAYS), dp_noise_rel=(0.5 * CAL["x"], 1.5 * CAL["x"]),
                          outlier_probability=CAL["p"])
V = S.V2(sigma_z=0.75, wander_sd=CAL["w"])


def main():
    out = HERE / "data"; out.mkdir(exist_ok=True)
    sea_path = SP / "solar/data/POWER_Point_Daily_20150101_20241231_029d00N_048d75E_LST.csv"
    sea = S.load_sea_temperature(sea_path)
    MT.V = V
    tabs, cls, hids, infos, pws = [], [], [], [], []
    for i, (seed, law) in enumerate(MT.PLANTS):
        t, c, h, info, pw = MT.one_plant(i, seed, law, sea, CFG, RO)
        tabs.append(t), cls.append(c), hids.append(h), infos.append(info), pws.append(pw)
        print(json.dumps(info))
    tab = pd.concat(tabs, ignore_index=True)
    assert not tab.isna().any().any()
    tab.to_csv(out / "table.csv", index=False, float_format="%.6g")
    pd.concat(cls, ignore_index=True).to_csv(out / "cleanings.csv", index=False)
    pd.concat(hids, ignore_index=True).to_csv(out / "hidden.csv", index=False, float_format="%.6g")
    pd.concat(pws, ignore_index=True).to_csv(out / "plantwide.csv", index=False, float_format="%.6g")
    (out / "plants.json").write_text(json.dumps({"plants": infos, "groups": MT.GROUPS, "sea_file": sea_path.name,
                                                 "measurement": {"screen_flow_exponent": 1.75, **CAL},
                                                 "sea_sha256": hashlib.sha256(sea_path.read_bytes()).hexdigest()}, indent=1))
    lines = [f"{hashlib.sha256((out / n).read_bytes()).hexdigest()}  {n}" for n in ("table.csv", "cleanings.csv")]
    (out / "table.sha256").write_text("\n".join(lines) + "\n")
    print("\n".join(lines)); print("rows", len(tab), "columns", len(tab.columns))


if __name__ == "__main__":
    main()
