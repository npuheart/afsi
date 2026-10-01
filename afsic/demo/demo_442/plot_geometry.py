"""Geometry figure for demo_442 (pressurized membrane), site asset generator."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = ("/Users/pengfei/Documents/GitHub/fdm-3d-v1-workspace/.local/"
       "mapengfei-glasgow.github.io/static/afsi/demo442-geometry.png")

fig, ax = plt.subplots(figsize=(6.2, 6.2))
ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, lw=1.6, ec="C7",
                           ls=(0, (6, 4))))
th = np.linspace(0, 2 * np.pi, 500)
ax.plot(0.5 + 0.25 * np.cos(th), 0.5 + 0.25 * np.sin(th), "-", color="C0",
        lw=2.2)
ax.annotate("", xy=(0.5 + 0.25 - 0.10, 0.5), xytext=(0.5 + 0.25 - 0.02, 0.5),
            arrowprops=dict(arrowstyle="->", color="C3", lw=1.6))
ax.annotate("", xy=(0.5 + 0.25 + 0.10, 0.5), xytext=(0.5 + 0.25 + 0.02, 0.5),
            arrowprops=dict(arrowstyle="->", color="C2", lw=1.6))
ax.text(0.60, 0.545, r"$p_i$", fontsize=12)
ax.text(0.815, 0.545, r"$p_o$", fontsize=12)
ax.text(0.5, 0.62, r"$R = 1/4$", fontsize=11, ha="center")
ax.text(0.02, 0.955, "periodic unit square (paper);\nwe use no-slip walls "
        "(deviation)", fontsize=9, color="0.35", va="top")
ax.plot(0.5, 0.5, "k+", ms=8)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.set_aspect("equal")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("pressurized circular membrane at equilibrium")
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print("saved", OUT)
