"""At what sea temperature does 40 % recovery (7 elements, 14 LMH, high-rejection membrane) cross a FilmTec limit?"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import vessel as V
fit = json.loads((Path(__file__).parent / "out" / "hrle_fit.json").read_text())
V.A20, V.B20 = fit["A20_LMH_per_bar"], fit["B20_LMH"]
out = {}
for T in (28.0, 30.0, 32.0, 33.0, 34.0, 35.0):
    v = V.operate(14.0, 0.40, 43.0, T, 7)
    out[T] = {"lead_flux": round(float(v["lead_flux"]), 2), "lead_rec": round(100 * float(v["lead_recovery"]), 2),
              "c_p": round(float(v["c_p_mgL"]), 1), "P_f": round(v["P_f"], 2), "sec_ro": round(v["sec_ro"], 3), "note": V.flags(v)}
print(json.dumps(out, indent=1))
(Path(__file__).parent / "out" / "threshold_40pct.json").write_text(json.dumps(out, indent=1))
