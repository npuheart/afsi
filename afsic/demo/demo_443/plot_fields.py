"""Field figure for demo_443 (compressed block): deformed solid coloured by J
and the fluid speed.  Site asset generator."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

X0b, X1b, Y0b, Y1b = 10.0, 30.0, 15.0, 25.0

src = sys.argv[1] if len(sys.argv) > 1 else "plot/snapshot_tfix.npz"
out = sys.argv[2] if len(sys.argv) > 2 else (
    "/Users/pengfei/Documents/GitHub/fdm-3d-v1-workspace/.local/"
    "mapengfei-glasgow.github.io/static/afsi/demo443-fields.png")
d = np.load(src)
M, MR = int(d["meta"][0]), int(d["meta"][1])
sc = d["solid_coords"].reshape(-1, 2)
sr = d["solid_ref"].reshape(-1, 2)
ij = d["solid_ij"]
n2x, n2y = 2 * M + 1, 2 * MR + 1
lookup = -np.ones((n2x, n2y), dtype=np.int64)
lookup[ij[:, 0], ij[:, 1]] = np.arange(len(ij))
assert (lookup >= 0).all()

rings = []
for i in range(M):
    for j in range(MR):
        ring = [(2 * i, 2 * j), (2 * i + 1, 2 * j), (2 * i + 2, 2 * j),
                (2 * i + 2, 2 * j + 1), (2 * i + 2, 2 * j + 2),
                (2 * i + 1, 2 * j + 2), (2 * i, 2 * j + 2),
                (2 * i, 2 * j + 1)]
        rings.append([lookup[a, b] for a, b in ring])
rings = np.array(rings)

Jx, Jv = d["J_x"][:, :2], d["J"]
Jcell = np.array([Jv[int(np.argmin(np.linalg.norm(
    Jx - sr[lookup[2 * i, 2 * j]], axis=1)))]
    for i in range(M) for j in range(MR)])

fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.0))
pc = PolyCollection(sc[rings], array=Jcell, cmap="coolwarm", edgecolors="k",
                    linewidths=0.25)
pc.set_clim(0.6, 1.4)
cb = plt.colorbar(pc, ax=axes[0], shrink=0.85)
cb.set_label("J")
axes[0].add_collection(pc)
axes[0].autoscale()
axes[0].set_aspect("equal")
axes[0].set_title(f"deformed block, J = [{Jv.min():.2f}, {Jv.max():.2f}]")

x = d["fluid_cx"] if "fluid_cx" in d else d["fluid_x"]
u = d["fluid_u"]
mag = np.linalg.norm(u, axis=1)
sp = axes[1].tricontourf(x[:, 0], x[:, 1], mag, levels=50, cmap="viridis")
plt.colorbar(sp, ax=axes[1], shrink=0.85).set_label(r"$|u|$")
axes[1].set_title(f"fluid speed, max = {mag.max():.2e}")

for ax in axes:
    ax.set_xlim(-1, 41)
    ax.set_ylim(-1, 41)
    ax.set_aspect("equal")
fig.suptitle(f"compressed block, M={M}, N=32")
fig.tight_layout()
fig.savefig(out, dpi=150)
print("saved", out)
