"""Compare the AFSI heart-tube run with the pyIB2d reference.

    python compare.py [ib2d_reference.npz] [out_dir]

Inputs: ``plot/afsi_result_g*.npz`` (main.py) and ``ib2d_reference.npz``
(run_reference.py).  The structure is a side-view tube: markers 0-154 are the
bottom wall, 155-309 the top wall; muscle band ``i`` connects markers
``i`` and ``i + 155`` (1 <= i <= 153).

Writes ``figures/compare_shapes.png`` (tube walls at six times),
``figures/compare_history.png`` (width profiles, mean width and net
through-flow), ``figures/compare_fields.png`` (vorticity and velocity
difference at the last common time) and ``figures/compare_table.csv``.
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
half = int(A["half"]) if "half" in A else A["X"].shape[1] // 2

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


def widths(Xm):
    """Width profile along the tube (153 muscle bands)."""
    return np.hypot(Xm[..., half + 1:2 * half - 1, 0] - Xm[..., 1:half - 1, 0],
                    Xm[..., half + 1:2 * half - 1, 1] - Xm[..., 1:half - 1, 1])


wa_prof, wr_prof = widths(Xa), widths(Xr)
wmean_a, wmean_r = wa_prof.mean(axis=1), wr_prof.mean(axis=1)
dX = np.array([np.linalg.norm(Xa[k] - Xr[k], axis=1).max() for k in range(len(t))])

# net through-flow: strip inside the tube, x-velocity
flow_a = flow_r = None
if "u" in R and "u" in A:
    ny, nx = R["u"].shape[1:3]
    xg = np.arange(nx) * float(R["dx"])
    yg = np.arange(ny) * float(R["dy"])
    strip = (np.abs(yg - 2.5) < 0.4)[:, None] & (xg > 1.5) & (xg < 3.5)
    flow_a = np.array([A["u"][ia][..., 0][strip].mean() for ia in ia_all])
    flow_r = np.array([R["u"][ir][..., 0][strip].mean() for ir in ir_all])

print(f"{'t':>6} {'|dX|max':>10} {'width AFSI':>11} {'width IB2d':>11} "
      f"{'flow AFSI':>10} {'flow IB2d':>10}")
rows = []
for k, tt in enumerate(t):
    fa = flow_a[k] if flow_a is not None else np.nan
    fr = flow_r[k] if flow_r is not None else np.nan
    rows.append((tt, dX[k], wmean_a[k], wmean_r[k], fa, fr))
    if k % 5 == 0 or k == len(t) - 1:
        print(f"{tt:6.3f} {dX[k]:10.3e} {wmean_a[k]:11.6f} {wmean_r[k]:11.6f} "
              f"{fa:10.5f} {fr:10.5f}")
with open(os.path.join(out, "compare_table.csv"), "w") as fh:
    fh.write("t,dXmax,width_afsi,width_ib2d,flow_afsi,flow_ib2d\n")
    for row in rows:
        fh.write(",".join(f"{v:.10g}" for v in row) + "\n")

# --- 1. tube walls at six times --------------------------------------------
sel = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
fig, axes = plt.subplots(1, len(sel), figsize=(2.5 * len(sel), 3.2))
for ax, tsel in zip(np.atleast_1d(axes), sel):
    k = int(np.argmin(abs(t - tsel)))
    for row, (D, col) in enumerate(((Xr, "k-"), (Xa, "r--"))):
        Xm = D[k]
        ax.plot(Xm[:half, 0], Xm[:half, 1], col, lw=2.0, label="IB2d" if row == 0 else "AFSI")
        ax.plot(Xm[half:, 0], Xm[half:, 1], col, lw=2.0)
    ax.set_title(f"t = {t[k]:.2f}")
    ax.set_aspect("equal")
    ax.set_xlim(0.8, 4.2)
    ax.set_ylim(1.5, 3.5)
np.atleast_1d(axes)[0].legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(out, "compare_shapes.png"), dpi=150)

# --- 2. histories -----------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
xt = np.linspace(1, 4, wa_prof.shape[1])
for k in range(0, len(t), max(1, len(t) // 8)):
    ax[0].plot(xt, wr_prof[k], "-", lw=1.2,
               color=plt.cm.viridis(k / max(1, len(t) - 1)))
    ax[0].plot(xt, wa_prof[k], "--", lw=1.0,
               color=plt.cm.viridis(k / max(1, len(t) - 1)))
ax[0].plot([], [], "k-", label="IB2d")
ax[0].plot([], [], "k--", label="AFSI")
ax[0].set_title("width profiles w(x)")
ax[0].set_xlabel("x")
ax[0].legend(fontsize=8)
ax[1].plot(t, wmean_r, "k-", lw=2, label="IB2d")
ax[1].plot(t, wmean_a, "r--", label="AFSI")
ax[1].set_title("mean tube width")
ax[1].set_xlabel("t")
ax[1].legend(fontsize=8)
if flow_a is not None:
    ax[2].plot(t, flow_r, "k-", lw=2, label="IB2d")
    ax[2].plot(t, flow_a, "r--", label="AFSI")
ax[2].set_title(r"mean $u_x$ inside tube (net pumping)")
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
    for a_, w, D, title in ((ax[0], wr_, R, "IB2d"), (ax[1], wa_, A, "AFSI")):
        c = a_.contourf(xg, yg, w, 40, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        Xm = D["X"][ir_all[k] if D is R else ia_all[k]]
        a_.plot(Xm[:half, 0], Xm[:half, 1], "k-", lw=1.0)
        a_.plot(Xm[half:, 0], Xm[half:, 1], "k-", lw=1.0)
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
