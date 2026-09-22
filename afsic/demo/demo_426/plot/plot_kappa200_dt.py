#!/usr/bin/env python3
"""kappa=200 fixed, dt reduced: does the plate displacement improve?"""
from pathlib import Path
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FACTORS = [0.2, 0.1, 0.05, 0.025]
H = 1.0 / 32.0
data = []
for f in FACTORS:
    p = Path(f"/tmp/k200_{f}.log")
    txt = p.read_text(errors="ignore")
    d = float(re.findall(r"plate max \|displacement\| from reference\s+([\d.eE+-]+)", txt)[-1])
    u = float(re.findall(r"fluid \|u\| on the plates \(should be 0\)\s+([\d.eE+-]+)", txt)[-1])
    data.append((f, f * H, d, u))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2))
dt = [x[1] for x in data]
disp = [x[2] for x in data]
pu = [x[3] for x in data]

a1.semilogx(dt, [d / (0.5 * H) for d in disp], "o-", lw=2, ms=8, color="tab:red")
a1.axhline(1.0, color="k", ls="--", lw=1.5, label=r"criterion $\delta \leq h/2$")
a1.set_xlabel("dt (s)")
a1.set_ylabel(r"$\delta_{\max} / (h/2)$")
a1.set_title(r"(a) kappa = 200 fixed, dt reduced 8x" "\n"
             r"$\delta$ does NOT improve (13.5 -> 13.7)")
a1.grid(alpha=0.3, which="both"); a1.legend()

a2.semilogx(dt, pu, "s-", lw=2, ms=8, color="tab:blue")
a2.set_xlabel("dt (s)")
a2.set_ylabel(r"fluid $|u|$ on the plates (cm/s)")
a2.set_title("(b) residual velocity on the plates"
             "\nalso unchanged (0.189 -> 0.194)")
a2.grid(alpha=0.3, which="both")

fig.suptitle("demo_426 (CGS, inflow ramp): fixed kappa=200, reducing dt",
             fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.94))
out = HERE / "kappa200_dt_sweep.png"
fig.savefig(out, dpi=170)
print("wrote", out)
for f, dtv, d, u in data:
    print(f"  DT_FACTOR={f:<6g} dt={dtv:<10.5g} d={d:.6g} ({d/(0.5*H):.2f} x h/2)  plate|u|={u:.4g}")
