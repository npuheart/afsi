#!/usr/bin/env python3
"""Plot the flow-history CSV produced by demo_424 main.py."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(here / "flow_history.csv")))
t = np.array([float(r["t"]) for r in rows])
q_gap = np.array([float(r["Q_gap"]) for r in rows])
q_up = np.array([float(r["Q_leak_up"]) for r in rows])
q_dn = np.array([float(r["Q_leak_dn"]) for r in rows])
umax = np.array([float(r["max_u"]) for r in rows])

fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

axes[0].plot(t, q_gap, "o-", color="tab:blue", lw=1.8, ms=4)
axes[0].axhline(0.0, color="k", lw=0.6)
axes[0].set_ylabel("Q_gap (m^2/s)")
axes[0].set_title("closed NY=45 S4: gap flux")
axes[0].grid(True, alpha=0.3)

axes[1].plot(t, q_up, "o-", color="tab:red", lw=1.8, ms=4, label="upstream")
axes[1].plot(t, q_dn, "s-", color="tab:orange", lw=1.8, ms=4, label="downstream")
axes[1].axhline(0.0, color="k", lw=0.6)
axes[1].set_ylabel("Q_leak (m^2/s)")
axes[1].set_title("lumen leakage flux at x_d +/- 4h")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(t, umax, "o-", color="tab:green", lw=1.8, ms=4)
axes[2].set_xlabel("t (s)")
axes[2].set_ylabel("max|u| (m/s)")
axes[2].set_title("global maximum velocity")
axes[2].grid(True, alpha=0.3)

for ax, y, name in ((axes[0], q_gap[-1], "Q_gap"),
                    (axes[1], q_up[-1], "Q_leak_up"),
                    (axes[2], umax[-1], "max|u|")):
    ax.annotate(f"{name} @1.0s = {y:.3e}",
                xy=(t[-1], y), xytext=(-10, 12), textcoords="offset points",
                ha="right", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

fig.tight_layout()
out = here / "flow_history_matplotlib.png"
fig.savefig(out, dpi=200)
print("wrote", out)

# Normalized decay: normalize by value at t=0.4 to see whether it is
# approaching a plateau.
i04 = int(np.argmin(np.abs(t - 0.4)))
fig2, ax = plt.subplots(figsize=(8, 5))
for y, lab, c in ((q_gap, "Q_gap", "tab:blue"),
                  (q_up, "Q_leak_up", "tab:red"),
                  (q_dn, "Q_leak_dn", "tab:orange"),
                  (umax, "max|u|", "tab:green")):
    ax.plot(t, y / y[i04], "o-", lw=1.8, ms=4, color=c, label=lab)
ax.axhline(1.0, color="k", lw=0.7)
ax.set_xlabel("t (s)")
ax.set_ylabel("value / value at t=0.4")
ax.set_title("normalized flow decay (closed NY=45 S4)")
ax.grid(True, alpha=0.3)
ax.legend()
fig2.tight_layout()
out2 = here / "flow_history_normalized_matplotlib.png"
fig2.savefig(out2, dpi=200)
print("wrote", out2)
