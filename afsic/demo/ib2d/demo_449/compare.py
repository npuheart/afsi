"""Compare the AFSI wobbly non-invariant-beam run with the pyIB2d reference.

    python compare.py [ib2d_reference.npz] [out_dir]

Inputs: ``plot/afsi_result_g*.npz`` (main.py) and ``ib2d_reference.npz``
(run_reference.py).  The beam is the marker set itself (no separate "bell");
both ends (markers 0 and N-1) are pinned by target points.

Writes ``figures/compare_shapes.png`` (beam outline at six times),
``figures/compare_history.png`` (mid-point height, marker deviation, final
profile), ``figures/compare_fields.png`` (vorticity and velocity difference at
the last common time) and ``figures/compare_table.csv``.
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
mid = int(A["mid_index"]) if "mid_index" in A else A["X"].shape[1] // 2

# common dump times (both runs dump every dt * print_dump)
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

dX = np.array([np.linalg.norm(Xa[k] - Xr[k], axis=1).max() for k in range(len(t))])
y_mid_a, y_mid_r = Xa[:, mid, 1], Xr[:, mid, 1]
y_max_a, y_max_r = Xa[:, :, 1].max(axis=1), Xr[:, :, 1].max(axis=1)

print(f"{'t':>7} {'|dX|max':>10} {'y_mid AFSI':>11} {'y_mid IB2d':>11} "
      f"{'y_max AFSI':>11} {'y_max IB2d':>11}")
rows = []
for k, tt in enumerate(t):
    rows.append((tt, dX[k], y_mid_a[k], y_mid_r[k], y_max_a[k], y_max_r[k]))
    if k % 10 == 0 or k == len(t) - 1:
        print(f"{tt:7.5f} {dX[k]:10.3e} {y_mid_a[k]:11.6f} {y_mid_r[k]:11.6f} "
              f"{y_max_a[k]:11.6f} {y_max_r[k]:11.6f}")
with open(os.path.join(out, "compare_table.csv"), "w") as fh:
    fh.write("t,dXmax,y_mid_afsi,y_mid_ib2d,y_max_afsi,y_max_ib2d\n")
    for row in rows:
        fh.write(",".join(f"{v:.10g}" for v in row) + "\n")

# --- 1. beam outlines at six times -----------------------------------------
sel = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
fig, axes = plt.subplots(1, len(sel), figsize=(2.5 * len(sel), 3.2), sharey=True)
for ax, tsel in zip(np.atleast_1d(axes), sel):
    k = int(np.argmin(abs(t - tsel)))
    ax.plot(Xr[k, :, 0], Xr[k, :, 1], "k-", lw=2.0, label="IB2d")
    ax.plot(Xa[k, :, 0], Xa[k, :, 1], "r--", lw=1.5, label="AFSI")
    ax.plot(Xr[k, [0, -1], 0], Xr[k, [0, -1], 1], "ks", ms=4)
    ax.set_title(f"t = {t[k]:.3f}")
    ax.set_aspect("equal")
    ax.set_xlim(0.15, 0.85)
    ax.set_ylim(0.40, 0.75)
np.atleast_1d(axes)[0].legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(out, "compare_shapes.png"), dpi=150)

# --- 2. histories -----------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
ax[0].plot(t, y_mid_r, "k-", lw=2, label="IB2d")
ax[0].plot(t, y_mid_a, "r--", label="AFSI")
ax[0].set_title("mid-point height $y_{mid}$")
ax[0].set_xlabel("t")
ax[0].legend(fontsize=8)
ax[1].semilogy(t[1:], dX[1:], "k-o", ms=3,
               label=r"$\max_k |X^{AFSI}-X^{IB2d}|$")
ax[1].axhline(R["dx"], color="gray", ls=":", label="grid spacing h")
ax[1].set_title("marker deviation")
ax[1].set_xlabel("t")
ax[1].legend(fontsize=8)
kf = len(t) - 1
ax[2].plot(Xr[kf, :, 0], Xr[kf, :, 1], "k-", lw=2, label=f"IB2d t={t[kf]:.3f}")
ax[2].plot(Xa[kf, :, 0], Xa[kf, :, 1], "r--", label=f"AFSI t={t[kf]:.3f}")
ax[2].set_title("final beam shape")
ax[2].set_aspect("equal")
ax[2].legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(out, "compare_history.png"), dpi=150)

# --- 3. fields at the last common time -------------------------------------
if "u" in R and "u" in A:
    k = len(t) - 1
    ua, ur = A["u"][ia_all[k]], R["u"][ir_all[k]]
    ny, nx = ur.shape[:2]
    xg = np.arange(nx) * float(R["dx"])
    yg = np.arange(ny) * float(R["dy"])

    def vort(u):
        duy = np.gradient(u[..., 0], yg, axis=0)
        dvx = np.gradient(u[..., 1], xg, axis=1)
        return dvx - duy

    wa, wr = vort(ua), vort(ur)
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    vmax = max(abs(wr).max(), abs(wa).max())
    for a_, w, title in ((ax[0], wr, "IB2d"), (ax[1], wa, "AFSI")):
        c = a_.contourf(xg, yg, w, 40, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        a_.plot(Xr[k, :, 0], Xr[k, :, 1], "k-", lw=1.2)
        a_.plot(Xa[k, :, 0], Xa[k, :, 1], "g--", lw=1.2)
        a_.set_title(f"{title}: vorticity, t = {t[k]:.3f}")
        a_.set_aspect("equal")
        fig.colorbar(c, ax=a_, shrink=0.8)
    c = ax[2].contourf(xg, yg, np.linalg.norm(ua - ur, axis=-1), 30, cmap="viridis")
    ax[2].set_title(r"$|u^{AFSI}-u^{IB2d}|$")
    ax[2].set_aspect("equal")
    fig.colorbar(c, ax=ax[2], shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "compare_fields.png"), dpi=150)

print(f"figures written to {out}")
