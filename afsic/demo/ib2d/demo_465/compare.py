"""Compare the AFSI moving circle with the pyIB2d reference, both schemes.

    python compare.py [out_dir]

Inputs, per interpolation scheme:

* linear: ``plot/afsi_result.npz``            and ``ib2d_reference_linear.npz``
* cubic:  ``plot_cubic/afsi_result.npz``      and ``ib2d_reference_cubic.npz``

(produced by ``python main.py`` / ``INTERP=cubic OUTPUT_PATH=./plot_cubic
python main.py`` and the two ``run_reference.py`` runs of readme.md).

Frame alignment: pyIB2d dumps when its 0-based counter ``cter`` is a positive
multiple of ``print_dump``, i.e. at the end of step ``cter + 1``, and stamps the
frame with ``cter * dt`` -- one step behind the state it wrote (upstream quirk).
``main.py`` dumps on exactly those steps, so frame ``k`` of the two codes is the
same step of the simulation and ``t_afsi[k] = t_ref[k] + dt``.

Writes ``figures/compare_shapes.png`` (the circle at several phases of the
cycle, both schemes), ``figures/compare_history.png`` (marker deviation and the
centre trajectory) and ``figures/compare_table.csv``.
"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

here = os.path.dirname(os.path.abspath(__file__))
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "figures")
os.makedirs(out, exist_ok=True)

SCHEMES = {"linear": (os.path.join(here, "plot", "afsi_result.npz"),
                      os.path.join(here, "ib2d_reference_linear.npz")),
           "cubic": (os.path.join(here, "plot_cubic", "afsi_result.npz"),
                     os.path.join(here, "ib2d_reference_cubic.npz"))}
h = 1.0 / 32                        # grid spacing of this example

data, rows = {}, []
for scheme, (fa, fr) in SCHEMES.items():
    A, R = np.load(fa), np.load(fr)
    n = min(len(A["t"]), len(R["t"]))
    t = A["t"][:n]
    Xa, Xr = A["X"][:n], R["X"][:n]
    dX = np.linalg.norm(Xa - Xr, axis=2).max(axis=1)
    ca, cr = Xa.mean(axis=1), Xr.mean(axis=1)
    lag = np.linalg.norm(Xa - A["Xa"][:n], axis=2).max(axis=1)
    data[scheme] = dict(t=t, Xa=Xa, Xr=Xr, dX=dX, ca=ca, cr=cr, lag=lag)
    print(f"[{scheme:6s}] frames AFSI {len(A['t'])} / IB2d {len(R['t'])}, "
          f"compared {n} (t = {t[0]:.5f} .. {t[-1]:.5f})")
    print(f"          marker deviation: max {dX.max():.3e} m = "
          f"{dX.max() / h:.3f} h, median {np.median(dX):.2e} m "
          f"(h = {h:g})")
    print(f"          centre trajectory: max |dc| = "
          f"{np.abs(ca - cr).max():.3e} m; circle-anchor lag "
          f"{lag.min():.2e}..{lag.max():.2e} m")
    for k in range(0, n, max(1, n // 8)):
        rows.append((scheme, t[k], dX[k], dX[k] / h, ca[k, 0], cr[k, 0],
                     ca[k, 1], cr[k, 1], lag[k]))
with open(os.path.join(out, "compare_table.csv"), "w") as fh:
    fh.write("scheme,t,dXmax,dXmax_over_h,centre_x_afsi,centre_x_ib2d,"
             "centre_y_afsi,centre_y_ib2d,lag_max\n")
    for row in rows:
        fh.write(f"{row[0]}," + ",".join(f"{v:.10g}" for v in row[1:]) + "\n")

# --- 1. the circle at several phases, both schemes --------------------------
nsel = 6
fig, axes = plt.subplots(2, nsel, figsize=(2.4 * nsel, 6.4))
for row, scheme in enumerate(SCHEMES):
    d = data[scheme]
    t, Xa, Xr = d["t"], d["Xa"], d["Xr"]
    sel = np.linspace(0.0, t[-1], nsel)
    for ax, tsel in zip(axes[row], sel):
        k = int(np.argmin(abs(t - tsel)))
        ax.plot(np.append(Xr[k, :, 0], Xr[k, 0, 0]),
                np.append(Xr[k, :, 1], Xr[k, 0, 1]), "k-", lw=1.8, alpha=0.55,
                label="IB2d")
        ax.plot(np.append(Xa[k, :, 0], Xa[k, 0, 0]),
                np.append(Xa[k, :, 1], Xa[k, 0, 1]), "r--", lw=1.1, label="AFSI")
        ax.set_title(f"{scheme}: t = {t[k]:.3f}")
        ax.set_aspect("equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[row][0].legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(out, "compare_shapes.png"), dpi=150)

# --- 2. histories -----------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for scheme, col in zip(SCHEMES, ("k", "b")):
    d = data[scheme]
    ax[0].semilogy(d["t"][1:], d["dX"][1:], col + "-o", ms=3, label=scheme)
    ax[1].plot(d["cr"][:, 0], d["cr"][:, 1], col + "-", lw=2, alpha=0.6,
               label=f"IB2d {scheme}")
    ax[1].plot(d["ca"][:, 0], d["ca"][:, 1], "r--", lw=1.0,
               label="AFSI (both)")
    ax[2].semilogy(d["t"][1:], d["lag"][1:], col + "-o", ms=3, label=scheme)
ax[0].axhline(h, color="gray", ls=":", label="grid spacing h")
ax[0].set_title("marker deviation vs IB2d")
ax[0].set_xlabel("t")
ax[0].legend(fontsize=8)
ax[1].set_title("circle centre trajectory (A-B-C-B cycle)")
ax[1].set_xlabel("x")
ax[1].set_ylabel("y")
ax[1].set_aspect("equal")
ax[1].legend(fontsize=8)
ax[2].set_title("circle-anchor lag $\\max|X-X_{anchor}|$")
ax[2].set_xlabel("t")
ax[2].legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(out, "compare_history.png"), dpi=150)

print(f"figures written to {out}")
