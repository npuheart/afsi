"""Field figure for demo_441 (Cook's membrane): deformed solid coloured by J
and the fluid speed around it.  Site asset generator."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

XA, XB = 3.25, 8.05
YA, YB, YC, YD = 3.5, 7.9, 9.5, 7.9

src = sys.argv[1] if len(sys.argv) > 1 else "plot/snapshot_fastM16v2.npz"
out = sys.argv[2] if len(sys.argv) > 2 else (
    "/Users/pengfei/Documents/GitHub/fdm-3d-v1-workspace/.local/"
    "mapengfei-glasgow.github.io/static/afsi/demo441-fields.png")
d = np.load(src)
M, MR = int(d["meta"][0]), int(d["meta"][1])
sc = d["solid_coords"].reshape(-1, 2)
sr = d["solid_ref"].reshape(-1, 2)
Jv, Jx = d["J"], d["J_x"][:, :2]

# structured index of every solid node
t = (sr[:, 0] - XA) / (XB - XA)
yb = YA + t * (YB - YA)
yt = YD + t * (YC - YD)
ii = np.rint(t * M).astype(int)
jj = np.rint((sr[:, 1] - yb) / (yt - yb) * MR).astype(int)
lookup = -np.ones((M + 1, MR + 1), dtype=int)
lookup[ii, jj] = np.arange(len(ii))
assert (lookup >= 0).all()

rings = []
for i in range(M):
    for j in range(MR):
        rings.append([lookup[i, j], lookup[i + 1, j], lookup[i + 1, j + 1],
                      lookup[i, j + 1]])
rings = np.array(rings)
Jcell = []
for k in range(M * MR):
    c = sr[rings[k]].mean(axis=0)
    Jcell.append(Jv[int(np.argmin(np.linalg.norm(Jx - c, axis=1)))])
Jcell = np.array(Jcell)

fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.2))
x, u = d["fluid_x"], d["fluid_u"]
mag = np.linalg.norm(u, axis=1)
sp = axes[0].tricontourf(x[:, 0], x[:, 1], mag, levels=50, cmap="viridis")
plt.colorbar(sp, ax=axes[0], shrink=0.85).set_label(r"$|u|$")
axes[0].set_title(f"fluid speed, max = {mag.max():.2e}")

pc = PolyCollection(sc[rings], array=Jcell, cmap="coolwarm", edgecolors="k",
                    linewidths=0.6)
pc.set_clim(0.7, 1.3)
cb = plt.colorbar(pc, ax=axes[1], shrink=0.85)
cb.set_label("J")
axes[1].add_collection(pc)
axes[1].autoscale()
axes[1].set_aspect("equal")
axes[1].set_title(f"deformed membrane, J = [{Jv.min():.3f}, {Jv.max():.3f}]")

for ax in axes:
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 13)
fig.suptitle("Cook's membrane, M=16, N=25 (fast protocol t=20 s)")
fig.tight_layout()
fig.savefig(out, dpi=150)
print("saved", out)
