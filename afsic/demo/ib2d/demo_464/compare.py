"""Compare the AFSI peristaltic pump with the pyIB2d reference.

    python compare.py [ib2d_reference.npz] [out_dir]

Inputs: ``plot/afsi_result_g*.npz`` (main.py) and ``ib2d_reference.npz``
(run_reference.py).  Compares the heart-tube markers (springs + invariant beams
+ target corners, with the resting lengths of the 76 cross-diameter pump springs
driven by the example's travelling-wave law).

Frame alignment: pyIB2d dumps when its 0-based counter ``cter`` is a positive
multiple of ``print_dump``, i.e. at the end of step ``cter + 1`` (and stamps the
frame with ``cter * dt``, one step behind the state it wrote); ``main.py`` dumps
on exactly those steps, so frame ``k`` of the two codes is the same step of the
simulation and ``t_afsi[k] = t_ref[k] + dt``.

The primary physical quantity is the pinch of the tube: the peristaltic wave
squeezes the cross-diameter springs from their resting length d = 1 down to
0.05, so the tube's width is the pumping signature.

Writes ``figures/compare_shapes.png``, ``figures/compare_history.png`` (widths,
marker deviation) and ``figures/compare_fields.png`` (vorticity and |du|).
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

f_afsi = os.path.join(here, "plot",
                      f"afsi_result_g{os.environ.get('GRAD_DIV', '2.5')}.npz")
for alt in ("afsi_result_g2.5.npz", "afsi_result.npz"):
    if not os.path.exists(f_afsi):
        f_afsi = os.path.join(here, "plot", alt)

R = np.load(f_ref)
A = np.load(f_afsi)
assert A["X"].shape[1] == R["X"].shape[1], "marker counts differ"

n = min(len(A["t"]), len(R["t"]))
assert n >= 2, "no common output times"
t = A["t"][:n]
Xa, Xr = A["X"][:n], R["X"][:n]

dX = np.linalg.norm(Xa - Xr, axis=2).max(axis=1)
h = float(R["dx"])
ya_min, ya_max = Xa[..., 1].min(axis=1), Xa[..., 1].max(axis=1)
yr_min, yr_max = Xr[..., 1].min(axis=1), Xr[..., 1].max(axis=1)
wa, wr = ya_max - ya_min, yr_max - yr_min
xa_c, xr_c = Xa[..., 0].mean(axis=1), Xr[..., 0].mean(axis=1)

print(f"frames: AFSI {len(A['t'])}, IB2d {len(R['t'])}, compared {n} "
      f"(t = {t[0]:.3f} .. {t[-1]:.3f}, dt = {float(2 / 2000):g})")
print(f"{'t':>7} {'dXmax':>9} {'dXmax/h':>8} {'width A/I':>17} "
      f"{'y-centre A/I':>19}")
rows = []
for k in range(n):
    rows.append((t[k], dX[k], dX[k] / h, wa[k], wr[k], xa_c[k], xr_c[k]))
    if k % max(1, n // 12) == 0 or k == n - 1:
        print(f"{t[k]:7.3f} {dX[k]:9.3e} {dX[k] / h:8.3e} "
              f"{wa[k]:8.4f}/{wr[k]:8.4f} "
              f"{ya_min[k] + wa[k] / 2:9.4f}/{yr_min[k] + wr[k] / 2:9.4f}")
with open(os.path.join(out, "compare_table.csv"), "w") as fh:
    fh.write("t,dXmax,dXmax_over_h,width_afsi,width_ib2d,ycentre_afsi,"
             "ycentre_ib2d\n")
    for row in rows:
        fh.write(",".join(f"{v:.10g}" for v in row) + "\n")

late = t >= 0.3        # after the wave's discontinuous start (see below)
print(f"\nmarker deviation : max {dX.max():.3e} m = {dX.max() / h:.2e} h "
      f"(h = {h:g}) at t = {t[int(np.argmax(dX))]:.3f}, median "
      f"{np.median(dX):.2e} m")
print(f"                 : for t >= 0.3 s the deviation stays below "
      f"{dX[late].max():.3e} m = {dX[late].max() / h:.2e} h (median "
      f"{np.median(dX[late]):.2e} m); the initial transient is expected because "
      f"the .m wave law is discontinuous at t = 0: sin(pi (x - x_M)) is +/-1 "
      f"over the tube, so half of the 76 pump springs *start* at their minimum "
      f"resting length (0.05)")
print(f"tube width       : AFSI {wa.min():.4f}..{wa.max():.4f}, IB2d "
      f"{wr.min():.4f}..{wr.max():.4f} (rest 1.0, wave minimum 0.05); "
      f"max |dw| {np.abs(wa - wr).max():.2e}")
print(f"pinch timing     : AFSI min at t={t[int(np.argmin(wa))]:.3f}, IB2d at "
      f"t={t[int(np.argmin(wr))]:.3f}; AFSI max at t={t[int(np.argmax(wa))]:.3f}, "
      f"IB2d at t={t[int(np.argmax(wr))]:.3f}")

# --- 1. tube at several times -----------------------------------------------
nsel = 6
sel = np.linspace(0.0, t[-1], nsel)
fig, axes = plt.subplots(1, nsel, figsize=(2.4 * nsel, 4.6))
pad = 0.15
lim = (Xr[..., 0].min() - pad, Xr[..., 0].max() + pad)
vlim = (Xr[..., 1].min() - pad, Xr[..., 1].max() + pad)
for ax, tsel in zip(np.atleast_1d(axes), sel):
    k = int(np.argmin(abs(t - tsel)))
    ax.plot(np.append(Xr[k, :, 0], Xr[k, 0, 0]),
            np.append(Xr[k, :, 1], Xr[k, 0, 1]), "k-", lw=1.8, alpha=0.55,
            label="IB2d")
    ax.plot(np.append(Xa[k, :, 0], Xa[k, 0, 0]),
            np.append(Xa[k, :, 1], Xa[k, 0, 1]), "r--", lw=1.1, label="AFSI")
    ax.set_title(f"t = {t[k]:.2f}")
    ax.set_aspect("equal")
    ax.set_xlim(*lim)
    ax.set_ylim(*vlim)
np.atleast_1d(axes)[0].legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(out, "compare_shapes.png"), dpi=150)

# --- 2. histories -----------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
ax[0].semilogy(t[1:], dX[1:], "k-o", ms=3, label="markers")
ax[0].axhline(h, color="gray", ls=":", label="grid spacing h")
ax[0].set_title("deviation vs IB2d")
ax[0].set_xlabel("t")
ax[0].legend(fontsize=8)
ax[1].plot(t, wr, "k-", lw=2, label="IB2d")
ax[1].plot(t, wa, "r--", label="AFSI")
ax[1].set_title("tube width (peristaltic pinch)")
ax[1].set_xlabel("t")
ax[1].legend(fontsize=8)
ax[2].plot(t, xr_c, "k-", lw=2, label="IB2d")
ax[2].plot(t, xa_c, "r--", label="AFSI")
ax[2].set_title("tube $x$-centroid")
ax[2].set_xlabel("t")
ax[2].legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(out, "compare_history.png"), dpi=150)

# --- 3. fields at the last common frame -------------------------------------
if "u" in R and "u" in A:
    # u/p are only sampled every FIELD_EVERY print_dumps, so the field
    # comparison uses the last frame of each file
    ua, ur = A["u"][-1], R["u"][-1]
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
    for a_, w, X, title in ((ax[0], wr_, Xr, "IB2d"), (ax[1], wa_, Xa, "AFSI")):
        c = a_.contourf(xg, yg, w, 40, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        a_.plot(np.append(X[k, :, 0], X[k, 0, 0]),
                np.append(X[k, :, 1], X[k, 0, 1]), "k-", lw=1.0)
        a_.set_title(f"{title}: vorticity, t = {t[k]:.2f}")
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
    print(f"velocity         : last-frame max|u| AFSI "
          f"{np.linalg.norm(ua, axis=-1).max():.3f}, IB2d "
          f"{np.linalg.norm(ur, axis=-1).max():.3f}; max|du| "
          f"{np.linalg.norm(ua - ur, axis=-1).max():.2e}")

print(f"figures written to {out}")
