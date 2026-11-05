"""Winter cycle for RO: what warming the winter feed does (high-rejection membrane, 7 elements, 40 %, 14 LMH, 43 g/L)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import vessel as V
fit = json.loads((Path(__file__).parent / "out" / "hrle_fit.json").read_text())
V.A20, V.B20 = fit["A20_LMH_per_bar"], fit["B20_LMH"]
out = {}
for T in (17.2, 20.0, 25.0, 28.0):              # 17.2 C: the February mean of the Gulf point, 2015-2024
    v = V.operate(14.0, 0.40, 43.0, T, 7)
    out[T] = {"P_f": round(v["P_f"], 2), "sec_ro": round(v["sec_ro"], 3), "c_p": round(float(v["c_p_mgL"]), 1),
              "lead_flux": round(float(v["lead_flux"]), 2), "lead_rec": round(100 * float(v["lead_recovery"]), 2),
              "ndp_tail": round(float(v["ndp_tail"]), 2), "note": V.flags(v)}
print(json.dumps(out, indent=1))
(Path(__file__).parent / "out" / "winter_cycle.json").write_text(json.dumps(out, indent=1))
