"""Geometry figure for demo_441 (Cook's membrane), site asset generator."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

OUT = ("/Users/pengfei/Documents/GitHub/fdm-3d-v1-workspace/.local/"
       "mapengfei-glasgow.github.io/static/afsi/demo441-geometry.png")

fig, ax = plt.subplots(figsize=(6.2, 6.0))
ax.add_patch(plt.Rectangle((0, 0), 13, 13, fill=False, lw=1.0, ec="0.6"))
A = (3.25, 3.5)
B = (8.05, 7.9)
C = (8.05, 9.5)
D = (3.25, 7.9)
ax.add_patch(Polygon([A, B, C, D], closed=True, fc="#cfe3ff", ec="C0",
                     lw=2.0))
ax.plot([A[0], D[0]], [A[1], D[1]], color="C3", lw=4.0, solid_capstyle="butt",
        label="clamped (penalty)")
n = 14
for k in range(n):
    x = B[0] + 0.45
    y = B[1] + (C[1] - B[1]) * (k + 0.5) / n
    ax.annotate("", xy=(x, y + 0.16), xytext=(x, y),
                arrowprops=dict(arrowstyle="->", color="C2", lw=1.4))
ax.plot([], [], color="C2", lw=1.4, label=r"traction $6.25\,\mathrm{dyn/cm}$")
ax.plot(C[0], C[1], "ko", ms=7)
ax.annotate(r"probe $\Delta Y$ at $(8.05,\,9.5)$", xy=C, xytext=(4.0, 10.6),
            arrowprops=dict(arrowstyle="->", lw=0.9))
ax.text(3.35, 5.6, r"$\Omega_0^{s}$", fontsize=13)
ax.text(0.15, 12.5, r"$\Omega$: fluid, $u=0$ on $\partial\Omega$",
        fontsize=10, color="0.4")
for p, lab, off in ((A, "A", (-0.55, -0.28)), (B, "B", (0.12, -0.28)),
                    (C, "C", (0.12, 0.12)), (D, "D", (-0.55, 0.12))):
    ax.text(p[0] + off[0], p[1] + off[1], lab, fontsize=11)
ax.annotate("", xy=(A[0], A[1] - 0.85), xytext=(D[0], D[1] - 0.85),
            arrowprops=dict(arrowstyle="<->", lw=0.8))
ax.text(3.6, 8.35, r"$4.4$ cm", fontsize=9)
ax.set_xlim(-0.6, 13.4)
ax.set_ylim(-0.6, 13.4)
ax.set_aspect("equal")
ax.set_xlabel("x (cm)")
ax.set_ylabel("y (cm)")
ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
ax.set_title("Cook's membrane in the $13\\times13$ cm fluid box")
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print("saved", OUT)
