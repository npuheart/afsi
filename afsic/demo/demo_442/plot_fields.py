"""Field figures for demo_442 (pressurized membrane): spurious vorticity and
velocity magnitude around the membrane, mirroring the paper's Fig. 4."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "plot/snapshot_paper.npz"
out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/membrane_fields.png"
d = np.load(src)
N, M_MEM, MFAC, DT, T_END, KAPPA, R, A0 = d["meta"]
CX = CY = 0.5

fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
ox, om = d["omega_x"], d["omega"]
v = np.abs(om).max()
sp = axes[0].tricontourf(ox[:, 0], ox[:, 1], om, levels=41,
                         cmap="RdBu_r", vmin=-v, vmax=v)
plt.colorbar(sp, ax=axes[0], shrink=0.85).set_label(r"$\omega$")
th = np.linspace(0, 2 * np.pi, 400)
axes[0].plot(CX + R * np.cos(th), CY + R * np.sin(th), "k-", lw=1.2)
axes[0].set_title(f"spurious vorticity (max |w| = {v:.3f})")

x, u = d["fluid_cx"] if "fluid_cx" in d else d["fluid_x"], d["fluid_u"]
mag = np.linalg.norm(u, axis=1)
sp = axes[1].tricontourf(x[:, 0], x[:, 1], mag, levels=41, cmap="viridis")
plt.colorbar(sp, ax=axes[1], shrink=0.85).set_label("|u|")
axes[1].plot(CX + R * np.cos(th), CY + R * np.sin(th), "w-", lw=1.0)
axes[1].set_title(f"velocity magnitude (max |u| = {mag.max():.2e})")

for ax in axes:
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
fig.suptitle(f"pressurized membrane, N={int(N)}, MFAC={MFAC:g}, t={T_END:g} s")
fig.tight_layout()
fig.savefig(out, dpi=140)
print("saved", out)
