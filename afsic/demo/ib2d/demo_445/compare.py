"""Compare the AFSI Hoover-jellyfish run with the IB2d reference solution.

    python compare.py [ib2d_reference.npz] [out_dir]

Inputs: ``plot/afsi_result.npz`` (main.py) and ``ib2d_reference.npz``
(run_reference.py, pyIB2d).  The bell outline is the first ``n_bell`` markers
(apex first, then the left arm, then the right arm); the remaining markers are
the flow blocker target points.

Writes ``figures/compare_shapes.png`` (bell + blockers at six times, IB2d vs
AFSI), ``figures/compare_history.png`` (bell centre of mass, apex, arm span,
marker deviation), ``figures/compare_fields.png`` (vorticity and velocity
difference at a selected time) and ``figures/compare_table.csv``.
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

# primary AFSI run: the gamma = 100 default; an existing gamma = 0 run is
# overlaid in the history figure as a secondary curve
f_afsi = sys.argv[3] if len(sys.argv) > 3 else os.path.join(here, "plot", "afsi_result_g100.npz")
if not os.path.exists(f_afsi):
    f_afsi = os.path.join(here, "plot", "afsi_result.npz")       # legacy name

R = np.load(f_ref)
A = np.load(f_afsi)
n_bell = int(A["n_bell"]) if "n_bell" in A else R["X"].shape[1] - 24
assert A["X"].shape[1] == R["X"].shape[1], "marker counts differ"

# common dump times (both runs dump every 0.01 s)
pairs = []
for ia, ta in enumerate(A["t"]):
    ir = int(np.argmin(abs(R["t"] - ta)))
    if abs(R["t"][ir] - ta) < 1e-8:
        pairs.append((ia, ir))
assert len(pairs) >= 2, "no common output times"
ia_all = [p[0] for p in pairs]
ir_all = [p[1] for p in pairs]
t = A["t"][ia_all]
Xa, Xr = A["X"][ia_all], R["X"][ir_all]


def outline(X):
    """Bell outline path: left arm tip -> apex -> right arm tip."""
    k = (n_bell + 1) // 2
    return np.vstack([X[:k][::-1], X[k:n_bell]])


tip_l = (n_bell + 1) // 2 - 1        # last point of the left arm
tip_r = n_bell - 1                   # last point of the right arm
com_a = Xa[:, :n_bell].mean(axis=1)
com_r = Xr[:, :n_bell].mean(axis=1)
span_a = np.linalg.norm(Xa[:, tip_l] - Xa[:, tip_r], axis=1)
span_r = np.linalg.norm(Xr[:, tip_l] - Xr[:, tip_r], axis=1)
dX = np.array([np.linalg.norm(Xa[k] - Xr[k], axis=1).max() for k in range(len(t))])

# table
print(f"{'t':>6} {'|dX|max':>10} {'com_y AFSI':>11} {'com_y IB2d':>11} "
      f"{'apex_y AFSI':>12} {'apex_y IB2d':>12} {'span AFSI':>10} {'span IB2d':>10}")
rows = []
for k, tt in enumerate(t):
    row = (tt, dX[k], com_a[k, 1], com_r[k, 1], Xa[k, 0, 1], Xr[k, 0, 1],
           span_a[k], span_r[k])
    rows.append(row)
    if k % 10 == 0 or k == len(t) - 1:
        print(f"{tt:6.3f} {dX[k]:10.3e} {com_a[k,1]:11.5f} {com_r[k,1]:11.5f} "
              f"{Xa[k,0,1]:12.5f} {Xr[k,0,1]:12.5f} {span_a[k]:10.5f} {span_r[k]:10.5f}")
with open(os.path.join(out, "compare_table.csv"), "w") as fh:
    fh.write("t,dXmax,com_y_afsi,com_y_ib2d,apex_y_afsi,apex_y_ib2d,span_afsi,span_ib2d\n")
    for row in rows:
        fh.write(",".join(f"{v:.10g}" for v in row) + "\n")

# --- 1. shapes: six dump times, IB2d vs AFSI -------------------------------
sel_times = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
y_lo = min(Xa[:, :n_bell, 1].min(), Xr[:, :n_bell, 1].min()) - 0.1
y_hi = max(Xa[:, :n_bell, 1].max(), Xr[:, :n_bell, 1].max()) + 0.1
fig, axes = plt.subplots(1, len(sel_times), figsize=(2.6 * len(sel_times), 3.4),
                         sharey=True)
for ax, tt in zip(np.atleast_1d(axes), sel_times):
    k = int(np.argmin(abs(t - tt)))
    Oa, Or = outline(Xa[k]), outline(Xr[k])
    ax.plot(Or[:, 0], Or[:, 1], "k-", lw=2.0, label="IB2d")
    ax.plot(Oa[:, 0], Oa[:, 1], "r--", lw=1.5, label="AFSI")
    ax.plot(Xr[k, n_bell:, 0], Xr[k, n_bell:, 1], ".", color="0.7", ms=2)
    ax.plot(Xa[k, n_bell:, 0], Xa[k, n_bell:, 1], ".", color="tab:orange", ms=2)
    ax.set_title(f"t = {t[k]:.2f}")
    ax.set_aspect("equal")
    ax.set_xlim(0.5, 2.5)
    ax.set_ylim(y_lo, y_hi)
np.atleast_1d(axes)[0].legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(out, "compare_shapes.png"), dpi=150)

# --- 2. histories -----------------------------------------------------------
fig, ax = plt.subplots(1, 4, figsize=(16, 3.4))
ax[0].plot(t, com_r[:, 1], "k-", lw=2, label="IB2d")
ax[0].plot(t, com_a[:, 1], "r--", label="AFSI")
ax[0].set_title("bell centre of mass $y$")
ax[1].plot(t, Xr[:, 0, 1], "k-", lw=2, label="IB2d")
ax[1].plot(t, Xa[:, 0, 1], "r--", label="AFSI")
ax[1].set_title("apex $y$")
ax[2].plot(t, span_r, "k-", lw=2, label="IB2d")
ax[2].plot(t, span_a, "r--", label="AFSI")
ax[2].set_title("arm-tip span")
ax[3].semilogy(t[1:], dX[1:], "k-", label=r"$\max_k|X^{AFSI}_k-X^{IB2d}_k|$")
ax[3].axhline(R["dx"], color="gray", ls=":", label="grid spacing h")
ax[3].set_title("marker deviation")
f_g0 = os.path.join(here, "plot", "afsi_result_g0.npz")
if os.path.exists(f_g0):
    A0 = np.load(f_g0)
    X0, t0 = A0["X"], A0["t"]
    ax[0].plot(t0, X0[:, :n_bell].mean(axis=1)[:, 1], color="tab:green", ls="-.",
               lw=1.2, label="AFSI $\\gamma$=0")
    ax[1].plot(t0, X0[:, 0, 1], color="tab:green", ls="-.", lw=1.2, label="AFSI $\\gamma$=0")
    ax[2].plot(t0, np.linalg.norm(X0[:, tip_l] - X0[:, tip_r], axis=1),
               color="tab:green", ls="-.", lw=1.2, label="AFSI $\\gamma$=0")
for a_ in ax:
    a_.set_xlabel("t")
    a_.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(out, "compare_history.png"), dpi=150)

# --- 3. fields at a selected time ------------------------------------------
if "u" in R and "u" in A:
    k = int(np.argmin(abs(t - 0.8)))
    ua, ur = A["u"][ia_all[k]], R["u"][ir_all[k]]
    ny, nx = ur.shape[:2]
    xg = np.arange(nx) * float(R["dx"])
    yg = np.arange(ny) * float(R["dy"])

    def vort(u):
        duy = np.gradient(u[..., 0], yg, axis=0)
        dvx = np.gradient(u[..., 1], xg, axis=1)
        return dvx - duy

    wa, wr = vort(ua), vort(ur)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    vmax = max(abs(wr).max(), abs(wa).max())
    for a_, w, title in ((ax[0], wr, "IB2d"), (ax[1], wa, "AFSI")):
        c = a_.contourf(xg, yg, w, 40, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        a_.plot(*outline(Xr[k]).T, "k-", lw=1.2)
        a_.plot(*outline(Xa[k]).T, "g--", lw=1.2)
        a_.set_title(f"{title}: vorticity, t = {t[k]:.2f}")
        a_.set_aspect("equal")
        fig.colorbar(c, ax=a_, shrink=0.8)
    c = ax[2].contourf(xg, yg, np.linalg.norm(ua - ur, axis=-1), 30, cmap="viridis")
    ax[2].set_title(r"$|u^{AFSI}-u^{IB2d}|$")
    ax[2].set_aspect("equal")
    fig.colorbar(c, ax=ax[2], shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "compare_fields.png"), dpi=150)

print(f"figures written to {out}")
