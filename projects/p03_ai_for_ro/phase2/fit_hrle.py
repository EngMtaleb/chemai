"""Fit A and B (at 20 C, our model's reference) so one element reproduces the SW30HRLE-440 datasheet test:
32 g/L NaCl, 800 psi (55.16 bar), 25 C, 8 % recovery -> 30.2 m3/d permeate, 99.80 % stabilized rejection."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import vessel as V

def element_at_test(A20, B20):
    V.A20, V.B20 = A20, B20
    lo, hi = 1.0, 40.0
    for _ in range(40):                                   # feed flow giving 8 % recovery
        q = (lo + hi) / 2
        v = V.pick(V.vessel(55.16, q, 32.0, 25.0, n=1), 0)
        lo, hi = (lo, q) if v["recovery"] < 0.08 else (q, hi)
    v = V.pick(V.vessel(55.16, (lo + hi) / 2, 32.0, 25.0, n=1), 0)
    return 24 * float(v["Q_p"]), 100 * (1 - float(v["c_p_mgL"]) / 1000 / 32.0)

A, B = 1.097, 0.0773
target_q, target_rej = 30.2, 99.80
for i in range(12):
    q, rej = element_at_test(A, B)
    A *= target_q / q
    B *= (100 - target_rej) / (100 - rej)
q, rej = element_at_test(A, B)
res = {"A20_LMH_per_bar": A, "B20_LMH": B, "check_permeate_m3d": q, "check_rejection_pct": rej,
       "phase1_A20": 1.097, "phase1_B20": 0.0773}
print(json.dumps(res, indent=1))
(Path(__file__).parent / "out" / "hrle_fit.json").write_text(json.dumps(res, indent=1))
