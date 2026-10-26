"""Build the RO decision table of the development plants (seed 1001, the plants Project 3
developed its forecast on) and of the real deposit's features (no scoring here)."""
import time
from pathlib import Path

import numpy as np

from chemai.config import RODepositConfig
from chemai.data.ro_deposit import download_deposit, load_daily, split_cycles
from chemai.data.ro_sim import generate_plants
from chemai.features.ro_decisions import real_decisions_table, sim_decisions

OUT = Path(__file__).parent / "results"

t0 = time.time()
dev = generate_plants(30, seed=1001)
df, seq = sim_decisions(dev, list(range(30)))
df.to_parquet(OUT / "ro_dev.parquet")
np.save(OUT / "ro_dev_seq.npy", seq)
print("dev", df.shape, round(time.time() - t0), "s")
