"""Compare the AFSI moving rubberband with the pyIB2d reference.

    python compare.py [ib2d_reference.npz] [out_dir]

Inputs: ``plot/afsi_result.npz`` (main.py) and ``ib2d_reference.npz``
(run_reference.py).

Frame alignment: the pyIB2d driver dumps when its 0-based step counter
``cter`` is a positive multiple of ``print_dump``, i.e. at the *end* of step
``cter + 1``, and stamps the frame with ``cter * dt`` -- one step behind the
state it writes (an upstream labelling quirk; MATLAB IB2d has the same one).
``main.py`` dumps on exactly those steps and stamps the true time, so AFSI
frame ``k`` and reference frame ``k`` are the same step of the simulation and
``t_afsi[k] = t_ref[k] + dt``.

Also reconstructs the prescribed anchor trajectory (the port of the example's
``update_Target_Point_Positions.m``) from the reference's own frames and
measures how far the markers trail their anchors -- the fluid-structure
quantity this example is about.

Writes ``figures/compare_shapes.png`` (band + anchors at several times),
``figures/compare_history.png`` (marker deviation, centroid, area, lag),
``figures/compare_fields.png`` (vorticity and |du|) and
``figures/compare_table.csv``.
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

sys.path.insert(0, here)
import ib2d_io  # noqa: E402

P = ib2d_io.read_input2d(os.path.join(here, "ib2d_input", "input2d"))
dt = float(P["dt"])
h = float(P["Lx"]) / int(P["Nx"])           # grid spacing

R = np.load(f_ref)
A = np.load(f_afsi)
n = min(len(A["t"]), len(R["t"]))
assert A["X"].shape[1] == R["X"].shape[1] == 64, "marker counts differ"

t = A["t"][:n]                  # true times of the AFSI (and reference) frames
Xa, Xr = A["X"][:n], R["X"][:n]
Xan = A["Xa"][:n]               # anchors as moved by the port of the .m file

dX = np.linalg.norm(Xa - Xr, axis=2).max(axis=1)
dcx = np.abs(Xa[:, :, 0].mean(axis=1) - Xr[:, :, 0].mean(axis=1))
dcy = np.abs(Xa[:, :, 1].mean(axis=1) - Xr[:, :, 1].mean(axis=1))


def area(X):
    x, y = X[..., 0], X[..., 1]
    return 0.5 * np.abs(np.sum(x * np.roll(y, -1, axis=1)
                               - y * np.roll(x, -1, axis=1), axis=1))


aa, ar = area(Xa), area(Xr)
lag_a = np.linalg.norm(Xa - Xan, axis=2).max(axis=1)
cx_a, cy_a = Xa[:, :, 0].mean(axis=1), Xa[:, :, 1].mean(axis=1)

# the anchors of the reference frames are not written by pyIB2d: rebuild the
# law of update_Target_Point_Positions.m from the frame times (step k * pd) and
# check that AFSI's anchors follow it exactly.
PERIOD, TRAVEL = 0.05, 0.5
SPEED = TRAVEL / (PERIOD / 2)
X0 = Xan[0].copy()


def anchor_displacement(nsteps):
    """Cumulative (dx, dy) of the .m update after ``nsteps`` applications."""
    dx = np.zeros(nsteps + 1)
    dy = np.zeros(nsteps + 1)
    ctime = 0.0
    for i in range(1, nsteps + 1):
        dy[i] = dy[i - 1] + (SPEED / 2) * dt
        dx[i] = dx[i - 1] + (SPEED * dt if np.remainder(ctime, PERIOD) < PERIOD / 2
                             else -SPEED * dt)
        ctime += dt
    return dx, dy


nsteps = np.rint(t / dt).astype(int)
dxl, dyl = anchor_displacement(int(nsteps[-1]))
Xan_law = X0[None, :, :] + np.stack([dxl[nsteps], dyl[nsteps]], axis=1)[:, None, :]
err_law = np.linalg.norm(Xan - Xan_law, axis=2).max()
lag_r = np.linalg.norm(Xr - Xan_law, axis=2).max(axis=1)

print(f"frames: AFSI {len(A['t'])}, IB2d {len(R['t'])}, compared {n} "
      f"(t = {t[0]:.5f} .. {t[-1]:.5f}, dt = {dt:g})")
print(f"{'t':>7} {'dXmax':>9} {'dXmax/h':>8} {'dCentroid':>10} "
      f"{'area A/I':>17} {'lag max':>8}")
rows = []
for k in range(n):
    rows.append((t[k], dX[k], dX[k] / h, max(dcx[k], dcy[k]),
                 aa[k], ar[k], lag_a[k]))
    if k % max(1, n // 12) == 0 or k == n - 1:
        print(f"{t[k]:7.4f} {dX[k]:9.3e} {dX[k] / h:8.3e} "
              f"{max(dcx[k], dcy[k]):10.3e} "
              f"{aa[k]:8.6f}/{ar[k]:8.6f} {lag_a[k]:8.2e}")
with open(os.path.join(out, "compare_table.csv"), "w") as fh:
    fh.write("t,dXmax,dXmax_over_h,dcentroid,area_afsi,area_ib2d,lag_max\n")
    for row in rows:
        fh.write(",".join(f"{v:.10g}" for v in row) + "\n")

print(f"\nmarker deviation : max {dX.max():.3e} m = {dX.max() / h:.2e} h "
      f"(h = {h:g}); typical {np.median(dX[1:]):.2e} m")
print(f"centroid         : max |dc| = {max(dcx.max(), dcy.max()):.3e} m")
print(f"enclosed area    : AFSI {aa.min():.6f}..{aa.max():.6f} (drift "
      f"{abs(aa[-1] - aa[0]):.2e}), IB2d {ar.min():.6f}..{ar.max():.6f} "
      f"(drift {abs(ar[-1] - ar[0]):.2e})")
print(f"anchor law       : AFSI anchors reproduce the .m recurrence to "
      f"{err_law:.2e} m")
print(f"marker lag       : AFSI {lag_a.min():.2e}..{lag_a.max():.2e} m, "
      f"IB2d {lag_r.min():.2e}..{lag_r.max():.2e} m "
      f"(the stiff tether trails its anchor)")

# --- 1. band + anchors at several times -------------------------------------
nsel = 6
sel = np.linspace(0.0, t[-1], nsel)
fig, axes = plt.subplots(1, nsel, figsize=(2.6 * nsel, 5.4))
lim = (Xr[..., 0].min() - 0.05, Xr[..., 0].max() + 0.05)
vlim = (Xr[..., 1].min() - 0.05, Xr[..., 1].max() + 0.05)
for ax, tsel in zip(np.atleast_1d(axes), sel):
    k = int(np.argmin(abs(t - tsel)))
    ax.plot(np.append(Xr[k, :, 0], Xr[k, 0, 0]),
            np.append(Xr[k, :, 1], Xr[k, 0, 1]), "k-", lw=2.0, alpha=0.55,
            label="IB2d")
    ax.plot(np.append(Xa[k, :, 0], Xa[k, 0, 0]),
            np.append(Xa[k, :, 1], Xa[k, 0, 1]), "r--", lw=1.2, label="AFSI")
    ax.plot(Xan[k, :, 0], Xan[k, :, 1], "b:", lw=0.8, label="anchors")
    ax.set_title(f"t = {t[k]:.4f}")
    ax.set_aspect("equal")
    ax.set_xlim(*lim)
    ax.set_ylim(*vlim)
np.atleast_1d(axes)[0].legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(out, "compare_shapes.png"), dpi=150)

# --- 2. histories -----------------------------------------------------------
fig, ax = plt.subplots(1, 4, figsize=(17, 3.4))
ax[0].semilogy(t[1:], dX[1:], "k-o", ms=3, label="markers")
ax[0].semilogy(t[1:], np.maximum(dcx[1:], 1e-16), "b-s", ms=3,
               label="centroid (x)")
ax[0].axhline(h, color="gray", ls=":", label="grid spacing h")
ax[0].axhline(np.median(dX[1:]), color="g", ls=":", label="median")
ax[0].set_title(r"deviation vs IB2d")
ax[0].set_xlabel("t")
ax[0].legend(fontsize=8)
ax[1].plot(t, Xr[:, :, 0].mean(axis=1), "k-", lw=2, label="IB2d")
ax[1].plot(t, cx_a, "r--", label="AFSI")
ax[1].plot(t, Xan[:, :, 0].mean(axis=1), "b:", label="anchors")
ax[1].set_title("centroid $x$ (right sweep, then left)")
ax[1].set_xlabel("t")
ax[1].legend(fontsize=8)
ax[2].plot(t, Xr[:, :, 1].mean(axis=1), "k-", lw=2, label="IB2d")
ax[2].plot(t, cy_a, "r--", label="AFSI")
ax[2].plot(t, Xan[:, :, 1].mean(axis=1), "b:", label="anchors")
ax[2].set_title("centroid $y$ (upward drift)")
ax[2].set_xlabel("t")
ax[2].legend(fontsize=8)
ax[3].plot(t, ar, "k-", lw=2, label="IB2d")
ax[3].plot(t, aa, "r--", label="AFSI($\\times$)")
ax[3].set_title("enclosed area + marker-anchor lag")
ax[3].set_xlabel("t")
axb = ax[3].twinx()
axb.semilogy(t, lag_a, "g-", lw=1, label="lag AFSI")
axb.semilogy(t, lag_r, "c--", lw=1, label="lag IB2d")
axb.set_ylabel("max $|X-X_{anchor}|$")
ax[3].legend(fontsize=8, loc="center right")
axb.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(out, "compare_history.png"), dpi=150)

# --- 3. fields at the last common frame -------------------------------------
if "u" in R and "u" in A:
    k = n - 1
    ua, ur = A["u"][k], R["u"][k]
    ny, nx = ur.shape[:2]
    xg = np.arange(nx) * float(R["dx"])
    yg = np.arange(ny) * float(R["dy"])

    def vort(u):
        duy = np.gradient(u[..., 0], yg, axis=0)
        dvx = np.gradient(u[..., 1], xg, axis=1)
        return dvx - duy

    wa_, wr_ = vort(ua), vort(ur)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    vmax = max(abs(wr_).max(), abs(wa_).max())
    for a_, w, X, title in ((ax[0], wr_, Xr, "IB2d"),
                            (ax[1], wa_, Xa, "AFSI")):
        c = a_.contourf(xg, yg, w, 40, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        a_.plot(np.append(X[k, :, 0], X[k, 0, 0]),
                np.append(X[k, :, 1], X[k, 0, 1]), "k-", lw=1.0)
        a_.plot(Xan[k, :, 0], Xan[k, :, 1], "b:", lw=0.8)
        a_.set_title(f"{title}: vorticity, t = {t[k]:.4f}")
        a_.set_aspect("equal")
        fig.colorbar(c, ax=a_, shrink=0.8)
    c = ax[2].contourf(xg, yg, np.linalg.norm(ua - ur, axis=-1), 30,
                       cmap="viridis")
    ax[2].set_title(r"$|u^{AFSI}-u^{IB2d}|$, "
                    f"max {np.linalg.norm(ua - ur, axis=-1).max():.2e}")
    ax[2].set_aspect("equal")
    fig.colorbar(c, ax=ax[2], shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "compare_fields.png"), dpi=150)
    print(f"velocity         : max|u| AFSI "
          f"{np.linalg.norm(ua, axis=-1).max():.3f}, IB2d "
          f"{np.linalg.norm(ur, axis=-1).max():.3f}; last-frame "
          f"max|du| {np.linalg.norm(ua - ur, axis=-1).max():.2e}")

print(f"figures written to {out}")
