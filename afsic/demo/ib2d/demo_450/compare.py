"""Compare the AFSI gravity cellular race with the pyIB2d reference.

    python compare.py [ib2d_reference.npz] [out_dir]

Inputs: ``plot/afsi_result_g*.npz`` (main.py) and ``ib2d_reference.npz``
(run_reference.py).  The two cells are markers 0-80 (A, left) and 81-161
(B, right); markers move with the fluid, the (invisible) ghost masses obey
their own ODE.

Writes ``figures/compare_shapes.png`` (both cells at six times),
``figures/compare_history.png`` (cell centres "the race", marker deviation),
``figures/compare_fields.png`` (vorticity and velocity difference) and
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

f_afsi = os.path.join(here, "plot", "afsi_result_g0.npz")
if not os.path.exists(f_afsi):
    f_afsi = os.path.join(here, "plot", "afsi_result.npz")

R = np.load(f_ref)
A = np.load(f_afsi)
assert A["X"].shape[1] == R["X"].shape[1], "marker counts differ"
nA = int(A["n_cellA"]) if "n_cellA" in A else A["X"].shape[1] // 2

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

comA_a, comA_r = Xa[:, :nA].mean(axis=1), Xr[:, :nA].mean(axis=1)
comB_a, comB_r = Xa[:, nA:].mean(axis=1), Xr[:, nA:].mean(axis=1)
dX = np.array([np.linalg.norm(Xa[k] - Xr[k], axis=1).max() for k in range(len(t))])

print(f"{'t':>6} {'|dX|max':>10} {'comA_y AFSI':>12} {'comA_y IB2d':>12} "
      f"{'comB_y AFSI':>12} {'comB_y IB2d':>12}")
rows = []
for k, tt in enumerate(t):
    rows.append((tt, dX[k], comA_a[k, 1], comA_r[k, 1], comB_a[k, 1], comB_r[k, 1]))
    if k % 5 == 0 or k == len(t) - 1:
        print(f"{tt:6.3f} {dX[k]:10.3e} {comA_a[k,1]:12.5f} {comA_r[k,1]:12.5f} "
              f"{comB_a[k,1]:12.5f} {comB_r[k,1]:12.5f}")
with open(os.path.join(out, "compare_table.csv"), "w") as fh:
    fh.write("t,dXmax,comAy_afsi,comAy_ib2d,comBy_afsi,comBy_ib2d\n")
    for row in rows:
        fh.write(",".join(f"{v:.10g}" for v in row) + "\n")

# --- 1. both cells at six times --------------------------------------------
sel = [0.0, 0.07, 0.14, 0.21, 0.28, 0.35]
fig, axes = plt.subplots(1, len(sel), figsize=(2.5 * len(sel), 3.2))
for ax, tsel in zip(np.atleast_1d(axes), sel):
    k = int(np.argmin(abs(t - tsel)))
    for X, X2, col, lab in ((Xr, Xa, "k-", "IB2d"), (Xa, Xr, "r--", "AFSI")):
        for sl in (slice(0, nA), slice(nA, None)):
            xs = X[k, sl, 0]
            ys = X[k, sl, 1]
            ax.plot(np.append(xs, xs[0]), np.append(ys, ys[0]), col, lw=1.5,
                    label=lab if (sl.start == 0 and col == "k-") else None)
    ax.set_title(f"t = {t[k]:.2f}")
    ax.set_aspect("equal")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.6, 1.0)
np.atleast_1d(axes)[0].legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(out, "compare_shapes.png"), dpi=150)

# --- 2. histories -----------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
ax[0].plot(t, comA_r[:, 1], "k-", lw=2, label="IB2d cell A")
ax[0].plot(t, comA_a[:, 1], "r--", label="AFSI cell A")
ax[0].plot(t, comB_r[:, 1], "k-", lw=2, alpha=0.6, label="IB2d cell B")
ax[0].plot(t, comB_a[:, 1], "r--", alpha=0.6, label="AFSI cell B")
ax[0].set_title("cell centres $y$ (the race)")
ax[0].set_xlabel("t")
ax[0].legend(fontsize=7)
ax[1].plot(t, comA_r[:, 0], "k-", lw=2, label="IB2d A")
ax[1].plot(t, comA_a[:, 0], "r--", label="AFSI A")
ax[1].plot(t, comB_r[:, 0], "k-", lw=2, alpha=0.6, label="IB2d B")
ax[1].plot(t, comB_a[:, 0], "r--", alpha=0.6, label="AFSI B")
ax[1].set_title("cell centres $x$")
ax[1].set_xlabel("t")
ax[1].legend(fontsize=7)
ax[2].semilogy(t[1:], dX[1:], "k-o", ms=3,
               label=r"$\max_k |X^{AFSI}-X^{IB2d}|$")
ax[2].axhline(R["dx"], color="gray", ls=":", label="grid spacing h")
ax[2].set_title("marker deviation")
ax[2].set_xlabel("t")
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

    wa_, wr_ = vort(ua), vort(ur)
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    vmax = max(abs(wr_).max(), abs(wa_).max())
    for a_, w, X, title in ((ax[0], wr_, Xr, "IB2d"), (ax[1], wa_, Xa, "AFSI")):
        c = a_.contourf(xg, yg, w, 40, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        for sl in (slice(0, nA), slice(nA, None)):
            a_.plot(X[k, sl, 0], X[k, sl, 1], "k-", lw=1.0)
        a_.set_title(f"{title}: vorticity, t = {t[k]:.2f}")
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
