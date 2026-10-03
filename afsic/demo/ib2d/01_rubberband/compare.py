"""Compare the AFSI port with the original IB2d run.

    python compare.py [afsi_result.npz] [ib2d_reference.npz] [out_dir]

Writes ``figures/compare_*.png`` and prints a table of differences at the IB2d
dump times.  Metrics:

* ``|dX|max``   max distance between corresponding Lagrangian points;
* ``area``      enclosed area of the band (IB volume leakage);
* ``e_u``       relative discrete L2 difference of the velocity on the grid;
* ``e_p``       relative discrete L2 difference of the (zero-mean) pressure.
"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

here = os.path.dirname(os.path.abspath(__file__))
f_afsi = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "plot", "afsi_result.npz")
f_ref = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "ib2d_reference.npz")
out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(here, "figures")
os.makedirs(out, exist_ok=True)

A = np.load(f_afsi)
R = np.load(f_ref)


def area(X):
    return 0.5 * abs(np.sum(X[:, 0] * np.roll(X[:, 1], -1) - np.roll(X[:, 0], -1) * X[:, 1]))


# match dump times
pairs = []
for ia, ta in enumerate(A["t"]):
    ir = np.argmin(abs(R["t"] - ta))
    if abs(R["t"][ir] - ta) < 1e-8:
        pairs.append((ia, ir))
if not pairs:
    sys.exit("no common output times")

rows = []
for ia, ir in pairs:
    Xa, Xr = A["X"][ia], R["X"][ir]
    dX = np.linalg.norm(Xa - Xr, axis=1).max()
    row = dict(t=A["t"][ia], dX=dX, area_a=area(Xa), area_r=area(Xr))
    if "u" in R and ir < len(R["u"]) and A["u"][ia].shape == R["u"][ir].shape:
        ua, ur = A["u"][ia], R["u"][ir]
        nr = np.sqrt((ur ** 2).sum())
        row["e_u"] = np.sqrt(((ua - ur) ** 2).sum()) / nr if nr > 0 else np.nan
        row["umax_a"], row["umax_r"] = np.abs(ua).max(), np.abs(ur).max()
        pa, pr = A["p"][ia], R["p"][ir]
        npr = np.sqrt((pr ** 2).sum())
        row["e_p"] = np.sqrt(((pa - pr) ** 2).sum()) / npr if npr > 0 else np.nan
    rows.append(row)

print(f"{'t':>7} {'|dX|max':>10} {'area AFSI':>10} {'area IB2d':>10} "
      f"{'e_u':>9} {'e_p':>9} {'max|u| AFSI':>12} {'max|u| IB2d':>12}")
for r in rows:
    print(f"{r['t']:7.3f} {r['dX']:10.3e} {r['area_a']:10.6f} {r['area_r']:10.6f} "
          f"{r.get('e_u', np.nan):9.3e} {r.get('e_p', np.nan):9.3e} "
          f"{r.get('umax_a', np.nan):12.4e} {r.get('umax_r', np.nan):12.4e}")
np.savetxt(os.path.join(out, "compare_table.csv"),
           np.array([[r["t"], r["dX"], r["area_a"], r["area_r"], r.get("e_u", np.nan),
                      r.get("e_p", np.nan), r.get("umax_a", np.nan), r.get("umax_r", np.nan)]
                     for r in rows]),
           delimiter=",", header="t,dXmax,area_afsi,area_ib2d,e_u,e_p,umax_afsi,umax_ib2d")

ia_all = [p[0] for p in pairs]
ir_all = [p[1] for p in pairs]
t = A["t"][ia_all]

# --- 1. band shapes at selected times
sel = [k for k, tt in enumerate(t) if np.any(np.isclose(tt, [0.0, 0.04, 0.1, 0.2, 0.5, t[-1]]))]
fig, axes = plt.subplots(1, len(sel), figsize=(2.6 * len(sel), 2.9), sharey=True)
axes = np.atleast_1d(axes)
for ax, k in zip(axes, sel):
    Xa, Xr = A["X"][ia_all[k]], R["X"][ir_all[k]]
    ax.plot(*np.vstack([Xr, Xr[:1]]).T, "k-", lw=2.0, label="IB2d")
    ax.plot(*np.vstack([Xa, Xa[:1]]).T, "r--", lw=1.4, label="AFSI")
    ax.set_title(f"t = {t[k]:.2f}")
    ax.set_aspect("equal")
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0.05, 0.95)
axes[0].legend(loc="lower left", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(out, "compare_shapes.png"), dpi=150)

# --- 2. time histories
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
Xa_t = A["X"][ia_all]
Xr_t = R["X"][ir_all]
ax[0].plot(t, Xr_t[:, 0, 0], "k-", label="IB2d  x(pt 1)")
ax[0].plot(t, Xa_t[:, 0, 0], "r--", label="AFSI  x(pt 1)")
nb = Xr_t.shape[1]
ax[0].plot(t, Xr_t[:, nb // 4, 1], "b-", label=f"IB2d  y(pt {nb//4+1})")
ax[0].plot(t, Xa_t[:, nb // 4, 1], "c--", label=f"AFSI  y(pt {nb//4+1})")
ax[0].set_xlabel("t")
ax[0].set_ylabel("position")
ax[0].legend(fontsize=8)
ax[1].plot(t, [r["area_r"] for r in rows], "k-", label="IB2d")
ax[1].plot(t, [r["area_a"] for r in rows], "r--", label="AFSI")
ax[1].set_xlabel("t")
ax[1].set_ylabel("enclosed area")
ax[1].legend(fontsize=8)
ax[2].semilogy(t[1:], [r["dX"] for r in rows][1:], "k-", label=r"$\max_k|X_k^{AFSI}-X_k^{IB2d}|$")
if "e_u" in rows[0]:
    ax[2].semilogy(t[1:], [r["e_u"] for r in rows][1:], "r-", label=r"$e_u$ (rel. $L^2$)")
    ax[2].semilogy(t[1:], [r["e_p"] for r in rows][1:], "b-", label=r"$e_p$ (rel. $L^2$)")
ax[2].axhline(R["dx"] if "dx" in R else 1 / 32, color="gray", ls=":", label="grid spacing h")
ax[2].set_xlabel("t")
ax[2].legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(out, "compare_history.png"), dpi=150)

# --- 3. velocity / pressure fields at a selected time
if "u" in R:
    k = min(range(len(t)), key=lambda i: abs(t[i] - 0.1))
    ua, ur = A["u"][ia_all[k]], R["u"][ir_all[k]]
    pa, pr = A["p"][ia_all[k]], R["p"][ir_all[k]]
    ny, nx = pr.shape
    xg = np.arange(nx) * float(R["dx"])
    yg = np.arange(ny) * float(R["dy"])
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    vmin, vmax = pr.min(), pr.max()
    for a_, pp, uu, title in ((ax[0], pr, ur, "IB2d"), (ax[1], pa, ua, "AFSI")):
        c = a_.contourf(xg, yg, pp, 30, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        a_.quiver(xg[::2], yg[::2], uu[::2, ::2, 0], uu[::2, ::2, 1], scale=40)
        a_.set_title(f"{title}: p and u, t = {t[k]:.2f}")
        a_.set_aspect("equal")
        fig.colorbar(c, ax=a_, shrink=0.8)
    c = ax[2].contourf(xg, yg, np.linalg.norm(ua - ur, axis=-1), 30, cmap="viridis")
    ax[2].set_title(r"$|u^{AFSI}-u^{IB2d}|$")
    ax[2].set_aspect("equal")
    fig.colorbar(c, ax=ax[2], shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "compare_fields.png"), dpi=150)

print(f"figures written to {out}")
