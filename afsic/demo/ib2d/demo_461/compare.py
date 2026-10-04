"""Compare the AFSI single coral with the pyIB2d reference.

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

The example's full run is 250000 steps, so only a twentieth of the 2.30 s
tentacle pulse is simulated (T = 0.115 s).  The anchor law is therefore also
validated *analytically over the whole pulse*: it is rebuilt from the .m
coefficients and its peak excursion, its periodicity and the fact that it
reproduces ``coral.vertex`` at the start of a cycle are reported.

Writes ``figures/compare_shapes.png`` (tentacles at several phases),
``figures/compare_history.png`` (marker deviation, x-extent, lag, prescribed
excursion), ``figures/compare_fields.png`` (vorticity and |du|) and
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
assert A["X"].shape[1] == R["X"].shape[1] == 90, "marker counts differ"

t = A["t"][:n]                  # true times of the AFSI (and reference) frames
Xa, Xr = A["X"][:n], R["X"][:n]
Xan = A["Xa"][:n]               # anchors as prescribed by the port of the .m

dX = np.linalg.norm(Xa - Xr, axis=2).max(axis=1)
dcx = np.abs(Xa[:, :, 0].mean(axis=1) - Xr[:, :, 0].mean(axis=1))
dcy = np.abs(Xa[:, :, 1].mean(axis=1) - Xr[:, :, 1].mean(axis=1))
lag_a = np.linalg.norm(Xa - Xan, axis=2).max(axis=1)
cx_a, cy_a = Xa[:, :, 0].mean(axis=1), Xa[:, :, 1].mean(axis=1)

# The anchors of the reference frames are not written by pyIB2d: rebuild the .m
# law from the example's own coefficients (the coral law is closed-form, unlike
# the pulsing heart's data-file interpolation) and check that AFSI's anchors
# follow it exactly.
F_FPS, S_PIX = 30.0, 1733.333333333333333333
TP_FRAME, TEND_FRAME = F_FPS * 2.25, F_FPS * 2.30
PERIOD_S = TEND_FRAME / F_FPS
XOFFSET, YOFFSET = 0.019, 0.25
XOFFSET2GRID = float(P["Lx"]) / 2
N_HALF = 45
QC = np.array([
    [-8.018203211249667e-13, 1.706310971452060e-10, -1.371526883228119e-8,
     5.116506115621489e-7, -8.613944795521634e-6, 4.951095471423292e-5,
     -1.237254488643602e-4],
    [-1.093988700839764e-10, 2.370737151797367e-8, -1.946684756263760e-6,
     7.447999061932255e-5, -1.285757196003425e-3, 7.184871148722453e-3,
     -2.357284935085283e-2],
    [-3.111491615328405e-9, 6.922752279229525e-7, -5.951659336196739e-5,
     2.461541908755476e-3, -4.830999119980220e-2, 3.293063238297988e-1,
     -1.893567487439901],
    [2.028150369106308e-8, -4.188949936239104e-6, 3.237380275082298e-4,
     -1.158763275070524e-2, 1.908244836752949e-1, -1.142372536532416,
     3.584246435337805],
])
X0 = np.loadtxt(os.path.join(here, "ib2d_input", "coral.vertex"), skiprows=1)
_xi = X0[:, 1] - YOFFSET
_left = np.arange(X0.shape[0]) < N_HALF


def anchors_from_law(times):
    """Prescribed anchor field X(t) of update_Target_Point_Positions.m."""
    out = np.empty((len(times),) + X0.shape)
    for k, ct in enumerate(times):
        fr = F_FPS * np.remainder(ct, PERIOD_S)
        if fr <= TP_FRAME:
            q = np.array([np.polyval(row, fr) for row in QC])
        else:
            w = (fr - TP_FRAME) / (TEND_FRAME - TP_FRAME)
            q = (np.array([np.polyval(row, TP_FRAME) for row in QC]) * (1.0 - w)
                 + QC[:, 6] * w)
        xt = (XOFFSET + q[3] / S_PIX - q[2] * _xi + q[1] * S_PIX * _xi ** 2
              - q[0] * S_PIX ** 2 * _xi ** 3)
        out[k, :, 0] = np.where(_left, -xt + XOFFSET2GRID, xt + XOFFSET2GRID)
        out[k, :, 1] = X0[:, 1]
    return out


# The anchors are advanced at the *start* of a step with that step's time, so a
# frame dumped at the end of step n carries the anchors of n - 1 applications.
# The time is the driver's *accumulated* clock (pyIB2d does current_time += dt),
# which is what both implementations pass to the law.
nsteps = np.rint(t / dt).astype(int)
ctimes = np.concatenate([[0.0], np.cumsum(np.full(int(nsteps[-1]), dt))])
Xan_law = anchors_from_law(ctimes[np.maximum(nsteps - 1, 0)])
err_law = np.linalg.norm(Xan - Xan_law, axis=2).max()
lag_r = np.linalg.norm(Xr - Xan_law, axis=2).max(axis=1)

# full-pulse checks, i.e. what the reduced run (a twentieth of a pulse) does
# not sample: peak excursion of the law and its exact periodicity
_ct = np.linspace(0.0, PERIOD_S, 2001)
_xc = anchors_from_law(_ct)[:, :, 0]
_peak = np.abs(_xc - X0[None, :, 0]).max()
_start = np.abs(anchors_from_law(np.array([0.0]))[0, :, 0] - X0[:, 0]).max()
_wrap = np.abs(anchors_from_law(np.array([PERIOD_S]))[0, :, 0]
               - anchors_from_law(np.array([0.0]))[0, :, 0]).max()

xa = Xa[:, :, 0].max(axis=1) - Xa[:, :, 0].min(axis=1)
xr = Xr[:, :, 0].max(axis=1) - Xr[:, :, 0].min(axis=1)
xn = Xan_law[:, :, 0].max(axis=1) - Xan_law[:, :, 0].min(axis=1)
excur = np.abs(Xan - Xan[0]).max(axis=(1, 2))

print(f"frames: AFSI {len(A['t'])}, IB2d {len(R['t'])}, compared {n} "
      f"(t = {t[0]:.5f} .. {t[-1]:.5f} of the {PERIOD_S:g} s pulse, dt = {dt:g})")
print(f"{'t':>7} {'dXmax':>9} {'dXmax/h':>8} {'dCentroid':>10} "
      f"{'x-extent A/I':>17} {'lag max':>8}")
rows = []
for k in range(n):
    rows.append((t[k], dX[k], dX[k] / h, max(dcx[k], dcy[k]),
                 xa[k], xr[k], lag_a[k]))
    if k % max(1, n // 12) == 0 or k == n - 1:
        print(f"{t[k]:7.4f} {dX[k]:9.3e} {dX[k] / h:8.3e} "
              f"{max(dcx[k], dcy[k]):10.3e} "
              f"{xa[k]:8.5f}/{xr[k]:8.5f} {lag_a[k]:8.2e}")
with open(os.path.join(out, "compare_table.csv"), "w") as fh:
    fh.write("t,dXmax,dXmax_over_h,dcentroid,xextent_afsi,xextent_ib2d,lag_max\n")
    for row in rows:
        fh.write(",".join(f"{v:.10g}" for v in row) + "\n")

print(f"\nmarker deviation : max {dX.max():.3e} m = {dX.max() / h:.2e} h "
      f"(h = {h:g}); typical {np.median(dX[1:]):.2e} m, and {np.median(dX):.2e} m "
      f"over the whole run")
print(f"centroid         : max |dc| = {max(dcx.max(), dcy.max()):.3e} m")
print(f"x-extent         : prescribed {xn.min():.5f}..{xn.max():.5f} m; "
      f"max|x-extent - prescribed| AFSI {np.abs(xa - xn).max():.2e}, "
      f"IB2d {np.abs(xr - xn).max():.2e}")
print(f"anchor law       : AFSI anchors reproduce the .m law to {err_law:.2e} m")
print(f"                 : over the full {PERIOD_S:g} s pulse the law spans "
      f"{_peak / h:.2f} h, reproduces coral.vertex to {_start:.2e} m, and is "
      f"periodic to {_wrap:.2e} m; this run covers "
      f"{excur[-1] / h:.2f} h of that")
print(f"marker lag       : AFSI {lag_a.min():.2e}..{lag_a.max():.2e} m, "
      f"IB2d {lag_r.min():.2e}..{lag_r.max():.2e} m "
      f"(the stiff tentacle trails its prescribed anchor)")

# --- 1. tentacles + anchors at several times --------------------------------
nsel = 6
sel = np.linspace(0.0, t[-1], nsel)
fig, axes = plt.subplots(1, nsel, figsize=(2.6 * nsel, 5.4))
pad = 0.02
lim = (min(Xr[..., 0].min(), Xan_law[..., 0].min()) - pad,
       max(Xr[..., 0].max(), Xan_law[..., 0].max()) + pad)
vlim = (Xr[..., 1].min() - pad, Xr[..., 1].max() + pad)
for ax, tsel in zip(np.atleast_1d(axes), sel):
    k = int(np.argmin(abs(t - tsel)))
    ax.plot(Xr[k, :, 0], Xr[k, :, 1], "k-", lw=2.0, alpha=0.55, label="IB2d")
    ax.plot(Xa[k, :, 0], Xa[k, :, 1], "r--", lw=1.2, label="AFSI")
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
ax[0].set_title(r"deviation vs IB2d")
ax[0].set_xlabel("t")
ax[0].legend(fontsize=8)
ax[1].plot(t, Xr[:, :, 0].mean(axis=1), "k-", lw=2, label="IB2d")
ax[1].plot(t, cx_a, "r--", label="AFSI")
ax[1].plot(t, Xan[:, :, 0].mean(axis=1), "b:", label="anchors")
ax[1].set_title("tentacle centroid $x$")
ax[1].set_xlabel("t")
ax[1].legend(fontsize=8)
ax[2].plot(t, xr, "k-", lw=2, label="IB2d")
ax[2].plot(t, xa, "r--", label="AFSI")
ax[2].plot(t, xn, "b:", label="prescribed")
ax[2].set_title("$x$-extent of the coral (tentacle bending)")
ax[2].set_xlabel("t")
ax[2].legend(fontsize=8)
ax[3].plot(t, excur / h, "k-", lw=2, label="anchor excursion")
ax[3].set_title(f"prescribed excursion (run covers {excur[-1] / h:.1f} h "
                f"of {_peak / h:.1f} h)")
ax[3].set_xlabel("t")
ax[3].set_ylabel("max $|X_{anchor}-X_{anchor}(0)|$ / h")
axb = ax[3].twinx()
axb.semilogy(t, lag_a, "g-", lw=1, label="lag AFSI")
axb.semilogy(t, lag_r, "c--", lw=1, label="lag IB2d")
axb.set_ylabel("max $|X-X_{anchor}|$")
ax[3].legend(fontsize=8, loc="upper left")
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
    for a_, w, X, title in ((ax[0], wr_, Xr, "IB2d"), (ax[1], wa_, Xa, "AFSI")):
        c = a_.contourf(xg, yg, w, 40, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        a_.plot(X[k, :, 0], X[k, :, 1], "k-", lw=1.0)
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
    print(f"velocity         : last-frame max|u| AFSI "
          f"{np.linalg.norm(ua, axis=-1).max():.3f}, IB2d "
          f"{np.linalg.norm(ur, axis=-1).max():.3f}; "
          f"max|du| {np.linalg.norm(ua - ur, axis=-1).max():.2e}; over the run "
          f"{np.linalg.norm(A['u'][:n], axis=-1).max():.3f} vs "
          f"{np.linalg.norm(R['u'][:n], axis=-1).max():.3f}")

print(f"figures written to {out}")
