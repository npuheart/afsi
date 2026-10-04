"""Compare the AFSI settling-particle run with the pyIB2d reference.

    python compare.py [ib2d_reference.npz] [out_dir]

Inputs: ``plot/afsi_result.npz`` (main.py) and ``ib2d_reference.npz``
(run_reference.py).

The example is a *small-signal* mass-model test: the ghost particle (M = 1e-3)
oscillates on its k = 1e3 spring at ~1000 rad/s while gravity makes the coupled
system creep downwards, so the marker moves only ~1e-5 m in T = 0.1 s.  The
figures therefore show the marker trajectory and its deviation, the marker
velocity, the ghost motion (AFSI only -- pyIB2d keeps it internal), and the
vorticity field at the final time.
"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

here = os.path.dirname(os.path.abspath(__file__))
f_ref = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "ib2d_reference.npz")
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "figures")
os.makedirs(out, exist_ok=True)

R = np.load(f_ref)
A = np.load(os.path.join(here, "plot", "afsi_result.npz"))
assert R["X"].shape[1] == A["X"].shape[1] == 1, "single-particle example"

pairs = []
for ia, ta in enumerate(A["t"]):
    ir = int(np.argmin(abs(R["t"] - ta)))
    if abs(R["t"][ir] - ta) < 1e-9:
        pairs.append((ia, ir))
ia_all = [p[0] for p in pairs]
ir_all = [p[1] for p in pairs]
t = A["t"][ia_all]
Xa = A["X"][ia_all, 0, :]
Xr = R["X"][ir_all, 0, :]
G = A["Xmass"][ia_all, 0, :]

dX = np.linalg.norm(Xa - Xr, axis=1)
h = float(R["dx"])
va = np.gradient(Xa[:, 1], t)

print(f"{'t':>7} {'y AFSI':>13} {'y IB2d':>13} {'dy':>11} {'|dX|':>11} "
      f"{'vy AFSI':>11} {'ghost y':>13}")
rows = []
for k in range(len(t)):
    rows.append((t[k], Xa[k, 1], Xr[k, 1], Xa[k, 1] - Xr[k, 1], dX[k],
                 va[k], G[k, 1]))
    if k % max(1, len(t) // 10) == 0 or k == len(t) - 1:
        print(f"{t[k]:7.4f} {Xa[k,1]:13.9f} {Xr[k,1]:13.9f} {Xa[k,1]-Xr[k,1]:+11.3e} "
              f"{dX[k]:11.3e} {va[k]:11.3e} {G[k,1]:13.9f}")
with open(os.path.join(out, "compare_table.csv"), "w") as fh:
    fh.write("t,y_afsi,y_ib2d,dy,dXmax,vy_afsi,ghost_y_afsi\n")
    for row in rows:
        fh.write(",".join(f"{v:.12g}" for v in row) + "\n")

print(f"marker displacement over T: AFSI {Xa[-1,1]-Xa[0,1]:+.4e} m, "
      f"IB2d {Xr[-1,1]-Xr[0,1]:+.4e} m "
      f"(ratio {(Xa[-1,1]-Xa[0,1])/(Xr[-1,1]-Xr[0,1]):.4f})")
print(f"marker deviation: max |dX| = {dX.max():.3e} m = {dX.max()/h:.2e} h; "
      f"final {dX[-1]:.3e} m")
print(f"ghost excursion (AFSI): y in [{G[:,1].min():.9f}, {G[:,1].max():.9f}] "
      f"-> amplitude {0.5*(G[:,1].max()-G[:,1].min()):.2e} m")

fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
ax[0].plot(t, Xr[:, 1], "k-", lw=2, label="IB2d")
ax[0].plot(t, Xa[:, 1], "r--", label="AFSI")
ax[0].set_title("marker height $y(t)$")
ax[0].set_xlabel("t")
ax[0].legend(fontsize=8)
ax[1].semilogy(t[1:], dX[1:], "k-o", ms=3, label=r"$|\Delta X|$")
ax[1].axhline(h, color="gray", ls=":", label="grid spacing h")
ax[1].set_title("marker deviation")
ax[1].set_xlabel("t")
ax[1].legend(fontsize=8)
ax[2].plot(t, Xa[:, 1], "r-", label="marker (AFSI)")
ax[2].plot(t, G[:, 1], "b--", label="ghost (AFSI, internal)")
ax[2].set_title("mass-spring-gravity oscillator")
ax[2].set_xlabel("t")
ax[2].legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(out, "compare_trajectory.png"), dpi=150)

if "u" in R and "u" in A:
    k = len(t) - 1
    ua, ur = A["u"][ia_all[k]], R["u"][ir_all[k]]
    ny, nx = ur.shape[:2]
    xg = np.arange(nx) * float(R["dx"])
    yg = np.arange(ny) * float(R["dy"])

    def vort(u):
        return (np.gradient(u[..., 1], xg, axis=1)
                - np.gradient(u[..., 0], yg, axis=0))

    wa_, wr_ = vort(ua), vort(ur)
    fig, ax = plt.subplots(1, 3, figsize=(3.2 * 3, 3.4))
    vmax = max(abs(wr_).max(), abs(wa_).max())
    for a_, w, X, title in ((ax[0], wr_, Xr, "IB2d"), (ax[1], wa_, Xa, "AFSI")):
        c = a_.contourf(xg, yg, w, 40, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        a_.plot(X[k, 0], X[k, 1], "ko", ms=4)
        a_.set_title(f"{title}: vorticity, t = {t[k]:.3f}")
        a_.set_aspect("equal")
        fig.colorbar(c, ax=a_, shrink=0.85)
    c = ax[2].contourf(xg, yg, np.linalg.norm(ua - ur, axis=-1), 30,
                       cmap="viridis")
    ax[2].set_title(r"$|u^{AFSI}-u^{IB2d}|$")
    ax[2].set_aspect("equal")
    fig.colorbar(c, ax=ax[2], shrink=0.85)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "compare_fields.png"), dpi=150)

print(f"figures written to {out}")
