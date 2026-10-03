"""Compare the AFSI single-porous rubberband with the pyIB2d reference.

    python compare.py [ib2d_reference.npz] [out_dir]

Inputs: ``plot/afsi_result_g*.npz`` (main.py) and ``ib2d_reference.npz``
(run_reference.py).

Writes ``figures/compare_shapes.png`` (the collapsing ring at several
times), ``figures/compare_history.png`` (enclosed area -- the primary
quantity -- and marker deviation), ``figures/compare_fields.png``
(vorticity) and ``figures/compare_table.csv``.
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

f_afsi = os.path.join(here, "plot", f"afsi_result_g{os.environ.get('GRAD_DIV', '100')}.npz")
if not os.path.exists(f_afsi):
    f_afsi = os.path.join(here, "plot", "afsi_result_g100.npz")
if not os.path.exists(f_afsi):
    f_afsi = os.path.join(here, "plot", "afsi_result_g0.npz")
if not os.path.exists(f_afsi):
    f_afsi = os.path.join(here, "plot", "afsi_result.npz")

R = np.load(f_ref)
A = np.load(f_afsi)
assert A["X"].shape[1] == R["X"].shape[1], "marker counts differ"

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


def area(X):
    return 0.5 * abs(np.dot(X[:, 0], np.roll(X[:, 1], -1))
                     - np.dot(X[:, 1], np.roll(X[:, 0], -1)))


dX = np.array([np.linalg.norm(Xa[k] - Xr[k], axis=1).max() for k in range(len(t))])
area_a = np.array([area(X) for X in Xa])
area_r = np.array([area(X) for X in Xr])

print(f"{'t':>6} {'|dX|max':>9} {'area AFSI':>10} {'area IB2d':>10} "
      f"{'area diff':>10}")
rows = []
for k, tt in enumerate(t):
    rows.append((tt, dX[k], area_a[k], area_r[k]))
    if k % max(1, len(t) // 12) == 0 or k == len(t) - 1:
        print(f"{tt:6.3f} {dX[k]:9.3e} {area_a[k]:10.5f} {area_r[k]:10.5f} "
              f"{area_a[k] - area_r[k]:+10.5f}")
with open(os.path.join(out, "compare_table.csv"), "w") as fh:
    fh.write("t,dXmax,area_afsi,area_ib2d\n")
    for row in rows:
        fh.write(",".join(f"{v:.10g}" for v in row) + "\n")

# --- 1. ring at several times -------------------------------------------------
nsel = 6
sel = np.linspace(0.0, t[-1], nsel)
fig, axes = plt.subplots(1, nsel, figsize=(2.4 * nsel, 3.2))
xlim = (Xr[..., 0].min() - 0.05, Xr[..., 0].max() + 0.05)
ylim = (Xr[..., 1].min() - 0.05, Xr[..., 1].max() + 0.05)
for ax, tsel in zip(np.atleast_1d(axes), sel):
    k = int(np.argmin(abs(t - tsel)))
    ax.plot(np.append(Xr[k, :, 0], Xr[k, 0, 0]),
            np.append(Xr[k, :, 1], Xr[k, 0, 1]), "k-", lw=1.6, label="IB2d")
    ax.plot(np.append(Xa[k, :, 0], Xa[k, 0, 0]),
            np.append(Xa[k, :, 1], Xa[k, 0, 1]), "r--", lw=1.2, label="AFSI")
    ax.set_title(f"t = {t[k]:.3f}")
    ax.set_aspect("equal")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
np.atleast_1d(axes)[0].legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(out, "compare_shapes.png"), dpi=150)

# --- 2. histories -------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
ax[0].plot(t, area_r, "k-", lw=2, label="IB2d")
ax[0].plot(t, area_a, "r--", label="AFSI")
ax[0].set_title("enclosed area (collapse)")
ax[0].set_xlabel("t")
ax[0].legend(fontsize=8)
ax[1].plot(t, area_r - area_a, "b-")
ax[1].set_title(r"area: IB2d $-$ AFSI")
ax[1].set_xlabel("t")
ax[2].semilogy(t[1:], dX[1:], "k-o", ms=3, label=r"$\max_k |X_k^{AFSI}-X_k^{IB2d}|$")
ax[2].axhline(R["dx"], color="gray", ls=":", label="grid spacing h")
ax[2].set_title("marker deviation")
ax[2].set_xlabel("t")
ax[2].legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(out, "compare_history.png"), dpi=150)

# --- 3. vorticity at the last common time --------------------------------------
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
    fig, ax = plt.subplots(1, 3, figsize=(3.2 * 3, 3.4))
    vmax = max(abs(wr_).max(), abs(wa_).max())
    for a_, w, X, title in ((ax[0], wr_, Xr, "IB2d"), (ax[1], wa_, Xa, "AFSI")):
        c = a_.contourf(xg, yg, w, 40, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        a_.plot(np.append(X[k, :, 0], X[k, 0, 0]),
                np.append(X[k, :, 1], X[k, 0, 1]), "k-", lw=1.0)
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
