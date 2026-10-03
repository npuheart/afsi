"""Compare the AFSI impedance pump with the pyIB2d reference.

    python compare.py [ib2d_reference.npz] [out_dir]

Inputs: ``plot/afsi_result_g*.npz`` (main.py) and ``ib2d_reference.npz``
(run_reference.py).  Compares the heart-tube markers (springs + invariant
beams + target corners, pinched by the driven pump springs) and the passive
tracers advected by the flow.

Writes ``figures/compare_shapes.png`` (tube + tracers at several times),
``figures/compare_history.png`` (marker deviation, tube width, tracer
drift), ``figures/compare_fields.png`` (vorticity / velocity difference)
and ``figures/compare_table.csv``.
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

f_afsi = os.path.join(here, "plot", "afsi_result_g0.npz")
if not os.path.exists(f_afsi):
    f_afsi = os.path.join(here, "plot", "afsi_result.npz")

R = np.load(f_ref)
A = np.load(f_afsi)
assert A["X"].shape[1] == R["X"].shape[1], "marker counts differ"
assert A["Xt"].shape[1] == R["tracers"].shape[1], "tracer counts differ"

pairs = []
for ia, ta in enumerate(A["t"]):
    ir = int(np.argmin(abs(R["t"] - ta)))
    if abs(R["t"][ir] - ta) < 1e-9:
        pairs.append((ia, ir))
assert len(pairs) >= 2, "no common output times"
ia_all = [p[0] for p in pairs]
ir_all = [p[1] for p in pairs]
t = A["t"][ia_all]
Xa, Xr = A["X"][ia_all], R["X"][ir_all]
Ta, Tr = A["Xt"][ia_all], R["tracers"][ir_all]

dX = np.array([np.linalg.norm(Xa[k] - Xr[k], axis=1).max() for k in range(len(t))])
# tracer trajectories are Lagrangian-chaotic: the pyIB2d reference itself
# diverges over ~1 s when dt changes 2x, so only robust statistics (median,
# p90) are meaningful; pointwise max is kept for the record.
dT = np.array([np.linalg.norm(Ta[k] - Tr[k], axis=1) for k in range(len(t))])
dXt = dT.max(axis=1)
dXt_med = np.median(dT, axis=1)
dXt_p90 = np.percentile(dT, 90, axis=1)
width_a = Xa[..., 1].max(axis=1) - Xa[..., 1].min(axis=1)
width_r = Xr[..., 1].max(axis=1) - Xr[..., 1].min(axis=1)
tmy_a, tmy_r = Ta[..., 1].mean(axis=1), Tr[..., 1].mean(axis=1)
tmx_a, tmx_r = Ta[..., 0].mean(axis=1), Tr[..., 0].mean(axis=1)

print(f"{'t':>7} {'|dX|max':>9} {'dXt med':>9} {'dXt p90':>9} {'width A/I':>15} "
      f"{'tracer-x com A/I':>19}")
rows = []
for k, tt in enumerate(t):
    rows.append((tt, dX[k], dXt_med[k], dXt_p90[k], dXt[k],
                 width_a[k], width_r[k], tmx_a[k], tmx_r[k]))
    if k % max(1, len(t) // 12) == 0 or k == len(t) - 1:
        print(f"{tt:7.3f} {dX[k]:9.2e} {dXt_med[k]:9.3f} {dXt_p90[k]:9.3f} "
              f"{width_a[k]:7.4f}/{width_r[k]:7.4f} "
              f"{tmx_a[k]:9.4f}/{tmx_r[k]:8.4f}")
with open(os.path.join(out, "compare_table.csv"), "w") as fh:
    fh.write("t,dXmax,dXt_median,dXt_p90,dXt_max,width_afsi,width_ib2d,"
             "tracerx_afsi,tracerx_ib2d\n")
    for row in rows:
        fh.write(",".join(f"{v:.10g}" for v in row) + "\n")

# --- 1. tube + tracers at several times -------------------------------------
nsel = 6
sel = np.linspace(0.0, t[-1], nsel)
fig, axes = plt.subplots(1, nsel, figsize=(2.6 * nsel, 5.4))
xlim = (Xr[..., 0].min() - 0.3, Xr[..., 0].max() + 0.3)
ylim = (min(Xr[..., 1].min(), Tr[..., 1].min()) - 0.3,
        max(Xr[..., 1].max(), Tr[..., 1].max()) + 0.3)
for ax, tsel in zip(np.atleast_1d(axes), sel):
    k = int(np.argmin(abs(t - tsel)))
    ax.plot(np.append(Xr[k, :, 0], Xr[k, 0, 0]),
            np.append(Xr[k, :, 1], Xr[k, 0, 1]), "k-", lw=1.6, label="IB2d")
    ax.plot(np.append(Xa[k, :, 0], Xa[k, 0, 0]),
            np.append(Xa[k, :, 1], Xa[k, 0, 1]), "r--", lw=1.2, label="AFSI")
    ax.plot(Tr[k, :, 0], Tr[k, :, 1], "k.", ms=2, alpha=0.6)
    ax.plot(Ta[k, :, 0], Ta[k, :, 1], "r.", ms=1.2, alpha=0.6)
    ax.set_title(f"t = {t[k]:.3f}")
    ax.set_aspect("equal")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
np.atleast_1d(axes)[0].legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(out, "compare_shapes.png"), dpi=150)

# --- 2. histories -----------------------------------------------------------
fig, ax = plt.subplots(1, 4, figsize=(17, 3.4))
ax[0].semilogy(t[1:], dX[1:], "k-o", ms=3, label=r"markers")
ax[0].semilogy(t[1:], np.maximum(dXt_med[1:], 1e-16), "b-s", ms=3,
               label=r"tracers (median)")
ax[0].semilogy(t[1:], np.maximum(dXt_p90[1:], 1e-16), "c-^", ms=3,
               label=r"tracers (p90)")
ax[0].axhline(R["dx"], color="gray", ls=":", label="grid spacing h")
ax[0].set_title(r"deviation vs IB2d")
ax[0].set_xlabel("t")
ax[0].legend(fontsize=8)
ax[1].plot(t, width_r, "k-", lw=2, label="IB2d")
ax[1].plot(t, width_a, "r--", label="AFSI")
ax[1].set_title("tube height (pinching)")
ax[1].set_xlabel("t")
ax[1].legend(fontsize=8)
ax[2].plot(t, tmy_r, "k-", lw=2, label="IB2d")
ax[2].plot(t, tmy_a, "r--", label="AFSI")
ax[2].set_title("tracer cloud $y$-mean")
ax[2].set_xlabel("t")
ax[2].legend(fontsize=8)
ax[3].plot(t, tmx_r, "k-", lw=2, label="IB2d")
ax[3].plot(t, tmx_a, "r--", label="AFSI")
ax[3].set_title("tracer cloud $x$-mean (bulk transport)")
ax[3].set_xlabel("t")
ax[3].legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(out, "compare_history.png"), dpi=150)

# --- 3. fields at the last common time -------------------------------------
if "u" in R and "u" in A:
    k = len(t) - 1
    ua = A["u"][int(np.argmin(abs(A["t_u"] - t[k])))]
    ur = R["u"][ir_all[k]]
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
    for a_, w, X, T, title in ((ax[0], wr_, Xr, Tr, "IB2d"),
                               (ax[1], wa_, Xa, Ta, "AFSI")):
        c = a_.contourf(xg, yg, w, 40, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        a_.plot(np.append(X[k, :, 0], X[k, 0, 0]),
                np.append(X[k, :, 1], X[k, 0, 1]), "k-", lw=1.0)
        a_.plot(T[k, :, 0], T[k, :, 1], "k.", ms=1.5)
        a_.set_title(f"{title}: vorticity, t = {t[k]:.3f}")
        a_.set_aspect("equal")
        fig.colorbar(c, ax=a_, shrink=0.8)
    c = ax[2].contourf(xg, yg, np.linalg.norm(ua - ur, axis=-1), 30,
                       cmap="viridis")
    ax[2].set_title(r"$|u^{AFSI}-u^{IB2d}|$")
    ax[2].set_aspect("equal")
    fig.colorbar(c, ax=ax[2], shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "compare_fields.png"), dpi=150)

print(f"figures written to {out}")
