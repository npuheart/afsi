"""Compare the AFSI passive-scalar solve with the pyIB2d reference.

    python compare.py [ib2d_reference.npz] [out_dir]

Inputs: ``plot/afsi_result.npz`` (main.py) and ``ib2d_reference.npz``
(run_reference.py).

The example is a pure-diffusion benchmark (the chamber wall is pinned, so the
fluid stays at rest), and a point source in 2D has the exact solution
``var(t) = var(0) + 2 k t`` per coordinate.  The figures therefore compare

* the concentration field itself (maps + centre-line profiles + difference),
* the per-coordinate variance against ``2 k t`` (the sharp quantitative check),
* the conserved total mass and the peak value,

and write ``figures/compare_table.csv`` with the full per-frame table.
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

f_afsi = os.path.join(here, "plot", "afsi_result.npz")

R = np.load(f_ref)
A = np.load(f_afsi)
assert "C" in R and "C" in A, "both files must contain the concentration frames"
assert A["C"].shape[1:] == R["C"].shape[1:], "grid shapes differ"

pairs = []
for ia, ta in enumerate(A["t"]):
    ir = int(np.argmin(abs(R["t"] - ta)))
    if abs(R["t"][ir] - ta) < 1e-9:
        pairs.append((ia, ir))
assert len(pairs) >= 2, "no common output times"
ia_all = [p[0] for p in pairs]
ir_all = [p[1] for p in pairs]
t = A["t"][ia_all]
Ca, Cr = A["C"][ia_all], R["C"][ir_all]
Xa, Xr = A["X"][ia_all], R["X"][ir_all]

Lx, Ly = R["dx"] * Ca.shape[2], R["dy"] * Ca.shape[1]
k_diff = float(A["k_diff"]) if "k_diff" in A else 0.025
xg = (np.arange(Ca.shape[2]) + 0.5) * R["dx"] - 0.5 * R["dx"]
yg = (np.arange(Ca.shape[1]) + 0.5) * R["dy"] - 0.5 * R["dy"]


def moments(c):
    """(total mass, peak, per-coordinate variance) of an (Ny, Nx) field."""
    tot = c.sum()
    x = (np.arange(c.shape[1])) * (Lx / c.shape[1])
    y = (np.arange(c.shape[0])) * (Ly / c.shape[0])
    cx, cy = c.sum(axis=0), c.sum(axis=1)
    mx, my = (cx * x).sum() / tot, (cy * y).sum() / tot
    vx = (cx * (x - mx) ** 2).sum() / tot
    vy = (cy * (y - my) ** 2).sum() / tot
    return tot, c.max(), vx, vy


def exact_field(t, cells, n_img=2):
    """Exact solution of the diffusion equation from the initial point sources.

    The initial 1.0 blobs are nodal point sources; the solution is the sum of
    the corresponding Gaussians *plus their periodic images* (the domain is a
    torus, so the images are physical, not an approximation).
    """
    ny, nx = Ca.shape[1], Ca.shape[2]
    X, Y = np.meshgrid(np.arange(nx) * R["dx"], np.arange(ny) * R["dy"],
                       indexing="ij")
    C = np.zeros((ny, nx))
    if t <= 0:
        return C
    for i0, j0 in cells:
        for n in range(-n_img, n_img + 1):
            for m in range(-n_img, n_img + 1):
                C += np.exp(-(((X - (i0 * R["dx"] + n * Lx)) ** 2
                               + (Y - (j0 * R["dy"] + m * Ly)) ** 2))
                            / (4 * k_diff * t))
    return C


# the initial nodal sources (grid cells where the example seeds concentration)
src_cells = np.argwhere(R["C"][0] > 0)

mA, mR = [moments(c) for c in Ca], [moments(c) for c in Cr]
mA = np.array(mA)
mR = np.array(mR)
v0 = mA[0, 2]
naive = v0 + 2 * k_diff * t                       # unbounded-domain formula
exact = np.array([moments(exact_field(tt, src_cells))[2] if tt > 0 else v0
                  for tt in t])                   # exact periodic solution

dC = np.array([np.abs(Ca[i] - Cr[i]).max() for i in range(len(t))])
l2 = np.array([np.linalg.norm(Ca[i] - Cr[i]) / np.linalg.norm(Cr[i])
               for i in range(len(t))])

print(f"{'t':>6} {'mass A/R':>17} {'Cmax A/R':>15} {'var AFSI':>12} "
      f"{'var IB2d':>12} {'var exact':>12} {'s0^2+2kt':>12} {'dCmax':>9} {'relL2':>9}")
rows = []
for j, tt in enumerate(t):
    rows.append((tt, mA[j, 0], mR[j, 0], mA[j, 1], mR[j, 1], mA[j, 2], mR[j, 2],
                 exact[j], naive[j], dC[j], l2[j]))
    if j % max(1, len(t) // 12) == 0 or j == len(t) - 1:
        print(f"{tt:6.3f} {mA[j,0]:8.5f}/{mR[j,0]:8.5f} "
              f"{mA[j,1]:7.5f}/{mR[j,1]:7.5f} {mA[j,2]:12.5e} {mR[j,2]:12.5e} "
              f"{exact[j]:12.5e} {naive[j]:12.5e} {dC[j]:9.2e} {l2[j]:9.3e}")
with open(os.path.join(out, "compare_table.csv"), "w") as fh:
    fh.write("t,mass_afsi,mass_ib2d,cmax_afsi,cmax_ib2d,var_afsi,var_ib2d,"
             "var_exact_periodic,var_sigma0_plus_2kt,dCmax,relL2\n")
    for row in rows:
        fh.write(",".join(f"{v:.10g}" for v in row) + "\n")

eA = (mA[:, 2] - exact) / exact
eR = (mR[:, 2] - exact) / exact
print(f"variance vs the exact periodic solution: AFSI max rel. err "
      f"{np.abs(eA).max():.2e} (final {eA[-1]:+.2e}), IB2d max rel. err "
      f"{np.abs(eR).max():.2e} (final {eR[-1]:+.2e})")
print(f"the naive unbounded formula sigma0^2+2kt is {(naive[-1]-exact[-1])/exact[-1]:+.2e} "
      f"off at t={t[-1]:.3f} (periodic-image effect, both codes follow the exact sum)")
print(f"marker deviation (pinned wall): max |dX| = {np.abs(Xa - Xr).max():.3e}")

# --- 1. field maps at several times -------------------------------------------
nsel = 4
sel = np.linspace(0.0, len(t) - 1, nsel).astype(int)
fig, axes = plt.subplots(3, nsel, figsize=(2.9 * nsel, 9.0))
vmax = max(Ca.max(), Cr.max())
for col, j in enumerate(sel):
    for row, (C, title) in enumerate(((Cr[j], "IB2d"), (Ca[j], "AFSI"))):
        ax = axes[row, col]
        c = ax.imshow(C, origin="lower", extent=(0, Lx, 0, Ly), cmap="inferno",
                      vmin=0.0, vmax=vmax)
        ax.set_title(f"{title}: C, t = {t[j]:.3f}")
        ax.set_aspect("equal")
        fig.colorbar(c, ax=ax, shrink=0.85)
    ax = axes[2, col]
    c = ax.imshow(Ca[j] - Cr[j], origin="lower", extent=(0, Lx, 0, Ly),
                  cmap="RdBu_r", vmin=-dC.max(), vmax=dC.max())
    ax.set_title(r"$C^{AFSI}-C^{IB2d}$")
    ax.set_aspect("equal")
    fig.colorbar(c, ax=ax, shrink=0.85)
fig.tight_layout()
fig.savefig(os.path.join(out, "compare_field.png"), dpi=150)

# --- 2. histories --------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
ax[0].plot(t, exact, "k-", lw=2, label=r"exact (periodic images)")
ax[0].plot(t, naive, "k:", lw=1.4, label=r"$\sigma_0^2 + 2kt$ (unbounded)")
ax[0].plot(t, mA[:, 2], "r--", label="AFSI")
ax[0].plot(t, mR[:, 2], "b:", label="IB2d")
ax[0].set_title(r"variance $\sigma^2(t)$")
ax[0].set_xlabel("t")
ax[0].legend(fontsize=7)
ax[1].semilogy(t[1:], np.abs(eA[1:]), "r-o", ms=3, label="AFSI")
ax[1].semilogy(t[1:], np.abs(eR[1:]), "b-o", ms=3, label="IB2d")
ax[1].semilogy(t[1:], np.abs(naive[1:] - exact[1:]) / exact[1:], "k:", lw=1.4,
               label=r"$\sigma_0^2+2kt$ vs exact")
ax[1].set_title(r"relative variance error vs the exact solution")
ax[1].set_xlabel("t")
ax[1].legend(fontsize=7)
ax[2].plot(t, mA[:, 0], "r--", label="mass AFSI")
ax[2].plot(t, mR[:, 0], "b:", label="mass IB2d")
ax2b = ax[2].twinx()
ax2b.semilogy(t[1:], dC[1:], "k-o", ms=3)
ax2b.set_ylabel(r"$\max|C^{AFSI}-C^{IB2d}|$")
ax[2].set_title("conserved mass (left) and field difference (right)")
ax[2].set_xlabel("t")
ax[2].legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(out, "compare_history.png"), dpi=150)

# --- 3. centre-line profiles ---------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
xs = np.arange(Ca.shape[2]) * (Lx / Ca.shape[2])
ys = np.arange(Ca.shape[1]) * (Ly / Ca.shape[1])
jmid = Ca.shape[1] // 2
for j in sel:
    ax[0].plot(xs, Cr[j, jmid], "b-", lw=1.6, alpha=0.8,
               label=f"IB2d t={t[j]:.3f}")
    ax[0].plot(xs, Ca[j, jmid], "r--", lw=1.2, label=f"AFSI t={t[j]:.3f}")
ax[0].set_title(f"centre-line C(x, y={ys[jmid]:.2f})")
ax[0].set_xlabel("x")
ax[0].legend(fontsize=7)
# azimuthally averaged radial profile (the blob is centred)
rr = np.hypot(xg[None, :] - 0.5, yg[:, None] - 0.5)
n_r = 40
edges = np.linspace(0.0, 0.5, n_r + 1)
mid = 0.5 * (edges[1:] + edges[:-1])
ridx = np.clip(np.digitize(rr.ravel(), edges) - 1, 0, n_r - 1)
for j in sel:
    for C, col, lab in ((Cr[j], "b-", "IB2d"), (Ca[j], "r--", "AFSI")):
        prof = np.bincount(ridx, weights=C.ravel(), minlength=n_r)
        cnt = np.bincount(ridx, minlength=n_r)
        ax[1].plot(mid, prof / np.maximum(cnt, 1), col, lw=1.4, alpha=0.8,
                   label=f"{lab} t={t[j]:.3f}")
ax[1].set_title("radially averaged C(r)")
ax[1].set_xlabel("r")
ax[1].legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(out, "compare_profiles.png"), dpi=150)

print(f"figures written to {out}")
