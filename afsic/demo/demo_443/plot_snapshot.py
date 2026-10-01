"""Quick-look plots for demo_443 snapshots: deformed block coloured by J."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

src = sys.argv[1] if len(sys.argv) > 1 else "plot/snapshot_b10v2.npz"
out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/block_J.png"
d = np.load(src)
M, MR = int(d["meta"][0]), int(d["meta"][1])
sc = d["solid_coords"].reshape(-1, 2)
sr = d["solid_ref"].reshape(-1, 2)
ij = d["solid_ij"]                       # structured index of every node
n2x, n2y = 2 * M + 1, 2 * MR + 1

# node lookup: structured (i, j) -> row index in the reshaped arrays
lookup = -np.ones((n2x, n2y), dtype=np.int64)
lookup[ij[:, 0], ij[:, 1]] = np.arange(len(ij))
assert (lookup >= 0).all(), "structured grid incomplete"

# cell outlines: 4 corners + 4 edge midpoints (P2)
rings = []
for ii in range(M):
    for jj in range(MR):
        ring = [(2 * ii, 2 * jj), (2 * ii + 1, 2 * jj), (2 * ii + 2, 2 * jj),
                (2 * ii + 2, 2 * jj + 1), (2 * ii + 2, 2 * jj + 2),
                (2 * ii + 1, 2 * jj + 2), (2 * ii, 2 * jj + 2),
                (2 * ii, 2 * jj + 1)]
        rings.append([lookup[a, b] for a, b in ring])
rings = np.array(rings)

# J lives on the P1 field: match its dof coordinates (J_x) to the vertex
# positions of each structured cell
Jx = d["J_x"][:, :2]
Jv = d["J"]
Jcell = []
for ii in range(M):
    for jj in range(MR):
        vxy = sr[lookup[2 * ii, 2 * jj]]
        k = int(np.argmin(np.linalg.norm(Jx - vxy, axis=1)))
        Jcell.append(Jv[k])
Jcell = np.array(Jcell)

fig, axes = plt.subplots(1, 4, figsize=(21, 5))
for ax, cor, cl, ti in ((axes[0], sr, None, "reference"),
                        (axes[1], sc, Jcell, "deformed, J"),
                        (axes[2], sc, None, "deformed outline")):
    if cl is None:
        pc = PolyCollection(cor[rings], facecolor="none",
                            edgecolors="k", linewidths=0.4)
    else:
        pc = PolyCollection(cor[rings], array=cl, cmap="coolwarm",
                            edgecolors="k", linewidths=0.15)
        pc.set_clim(0.3, 1.5)
        plt.colorbar(pc, ax=ax, shrink=0.85).set_label("J")
    ax.add_collection(pc)
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_title(ti)

ax = axes[3]
U = d["fluid_u"]
sp = ax.scatter(d["fluid_x"][:, 0], d["fluid_x"][:, 1],
                c=np.linalg.norm(U, axis=1), s=2, cmap="viridis")
plt.colorbar(sp, ax=ax, shrink=0.85).set_label("|u|")
ax.set_aspect("equal")
ax.set_title("fluid |u|")
for ax in axes:
    ax.set_xlim(-1, 41)
    ax.set_ylim(-1, 41)
fig.tight_layout()
fig.savefig(out, dpi=130)
print("saved", out)
disp = sc - sr
on_top = np.isclose(sr[:, 1], 25.0)
print("J range:", d["J"].min(), d["J"].max())
print("top dY [min,max]:", disp[on_top, 1].min(), disp[on_top, 1].max())
i = int(np.argmin(d["J"]))
print("J_min node ref:", sr[i], "cur:", sc[i])
