"""Geometry figure for demo_443 (compressed block), site asset generator."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = ("/Users/pengfei/Documents/GitHub/fdm-3d-v1-workspace/.local/"
       "mapengfei-glasgow.github.io/static/afsi/demo443-geometry.png")

fig, ax = plt.subplots(figsize=(6.2, 6.2))
ax.add_patch(Rectangle((0, 0), 40, 40, fill=False, lw=1.0, ec="0.6"))
ax.add_patch(Rectangle((10, 15), 20, 10, fc="#cfe3ff", ec="C0", lw=2.0))
ax.plot([10, 30], [15, 15], color="C3", lw=4.0, solid_capstyle="butt",
        label="zero vertical displacement (penalty)")
ax.plot([10, 15], [25, 25], color="C1", lw=4.0, solid_capstyle="butt")
ax.plot([25, 30], [25, 25], color="C1", lw=4.0, solid_capstyle="butt",
        label="zero horizontal displacement (penalty)")
n = 9
for k in range(n):
    x = 15 + 10 * (k + 0.5) / n
    ax.annotate("", xy=(x, 24.05), xytext=(x, 25.35),
                arrowprops=dict(arrowstyle="->", color="C2", lw=1.4))
ax.plot([], [], color="C2", lw=1.4,
        label=r"traction $200\,\mathrm{dyn/cm}$ over $10$ cm")
ax.plot(20, 25, "ko", ms=7)
ax.annotate(r"probe $\Delta Y$ at $(20,\,25)$", xy=(20, 25), xytext=(25.5, 27.7),
            arrowprops=dict(arrowstyle="->", lw=0.9))
ax.text(19.1, 19.4, r"$\Omega_0^{s}$", fontsize=13)
ax.text(14.9, 14.45, r"$20$ cm", fontsize=9, ha="center")
ax.text(29.6, 19.8, r"$10$ cm", fontsize=9, rotation=90, va="center")
ax.text(0.4, 38.9, r"$\Omega$: fluid, $u=0$ on $\partial\Omega$",
        fontsize=10, color="0.4")
ax.set_xlim(-1, 41)
ax.set_ylim(-1, 41)
ax.set_aspect("equal")
ax.set_xlabel("x (cm)")
ax.set_ylabel("y (cm)")
ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
ax.set_title("compressed block in the $40\\times40$ cm fluid box")
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print("saved", OUT)
