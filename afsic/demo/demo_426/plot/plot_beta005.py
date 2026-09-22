#!/usr/bin/env python3
"""Plot the beta=0.05 pure-tether run: exponential divergence of the plate.

The plate is advected by the interpolated fluid velocity and held only by the
tether F = -beta (X - X_ref); the force is spread by AFSI's distributor, whose
integration weight w is hard-coded to 1 (one full cell per marker).
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "plot" / "N32_ipcs_smoke" / "history.csv"

rows = list(csv.DictReader(open(SRC)))
t = np.array([float(r["t"]) for r in rows])
umax = np.array([float(r["max_u"]) for r in rows])
disp = np.array([float(r["plate_disp"]) for r in rows])
h = 1.0 / 32.0            # grid size

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

ax = axes[0]
ax.semilogy(t, disp, lw=1.8, color="tab:red", label="plate displacement")
ax.semilogy(t, umax, lw=1.8, color="tab:blue", label=r"$\max|u|$")
ax.axhline(h / 2, color="k", ls="--", lw=1.2,
           label=r"paper tolerance: $\delta \leq h/2$")
ax.axhline(1.1547, color="grey", ls=":", lw=1.2, label="channel width D")
ax.set_xlabel("t (s)")
ax.set_ylabel("magnitude")
ax.set_title(r"(a) $\beta=0.05$, pure tether, $w$ hard-coded to 1")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=9)

ax = axes[1]
# growth rate
mask = disp > 0
rate = np.gradient(np.log(disp[mask]), t[mask])
ax.plot(t[mask], rate, lw=1.8, color="tab:purple")
ax.axhline(0.0, color="k", lw=0.8)
ax.set_xlabel("t (s)")
ax.set_ylabel(r"d$\log\delta$/dt  (1/s)")
ax.set_title("(b) growth rate: positive and rising = exponential blow-up")
ax.grid(alpha=0.3)

fig.suptitle("demo_426 — pure tether at beta=0.05 diverges (explicit, w = 1)",
             fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.94))
out = HERE / "beta005_divergence.png"
fig.savefig(out, dpi=170)
print("wrote", out)
