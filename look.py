import glob, os, pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load(p):
    d = pd.read_csv(p, sep=';')
    d.columns = [c.strip().lower() for c in d.columns]
    return d

files = []
for cat in ["Stiction", "Tuning"]:
    for f in sorted(glob.glob(f"data/sacac/{cat}/**/*.csv", recursive=True)):
        files.append((cat, f))

print(f"{len(files)} loops\n")
fig, ax = plt.subplots(2, len(files), figsize=(4.2*len(files), 8))
if len(files) == 1: ax = ax.reshape(2, 1)

for i, (cat, f) in enumerate(files):
    d = load(f)
    n = min(len(d), 800)
    pv, op = d.pv.values[:n], d.op.values[:n]
    ax[0, i].plot(pv, lw=.8, label="PV"); ax[0, i].plot(op, lw=.8, label="OP")
    ax[0, i].set_title(f"{cat}\n{os.path.basename(f)[:26]}", fontsize=8)
    ax[0, i].legend(fontsize=6)
    ax[1, i].plot(op, pv, lw=.6, color="#0E7C99")
    ax[1, i].set_xlabel("OP"); ax[1, i].set_ylabel("PV")
    print(f"{cat:<10} {os.path.basename(f)[:40]:<42} PV {pv.min():.1f}-{pv.max():.1f}")

plt.tight_layout(); plt.savefig("loops.png", dpi=110)
print("\nsaved loops.png")
