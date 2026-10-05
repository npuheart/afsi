"""Compare the AFSI electromechanical-pumping run with the pyIB2d reference.

    python compare.py [ib2d_reference.npz] [out_dir]

Inputs: ``plot/afsi_result.npz`` (main.py) and ``ib2d_reference.npz``
(run_reference.py).  Frame alignment: the driver's pre-loop frame is the
initial state (step 0) and frame ``k >= 1`` is the state at step
``k * print_dump + 1`` (its VTK time stamp is one step behind the state it
wrote -- an upstream labelling quirk); main.py dumps on exactly those steps.

Metrics: marker deviation (in grid spacings ``h``), the cross-tube diameter
(the physical output of the pumping), and the *applied electrophysiology* --
the stiffness ``(16 v)^4`` written into the 78 cross-tube springs, which is the
only channel through which the FitzHugh-Nagumo potential acts.

Writes ``figures/compare_shapes.png``, ``figures/compare_history.png`` and
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

R = np.load(f_ref)
A = np.load(f_afsi)
assert A["X"].shape[1] == R["X"].shape[1], "marker counts differ"
dt = float(R["dt"])
h = float(R["dx"])
nb = A["X"].shape[1]
half = int(A["half"])
start, end = int(A["ephys_start"]), int(A["ephys_end"])

steps_a = np.rint(A["t"] / dt).astype(int)
steps_r = np.rint(R["t"] / dt).astype(int)
keep = [k for k in range(min(len(steps_a), len(steps_r)))
        if steps_a[k] == steps_r[k] + (0 if k == 0 else 1)]
assert keep, f"no aligned frames: AFSI {steps_a} vs reference {steps_r}"
ia, ir = np.array(keep), np.array(keep)
t = A["t"][ia]
Xa, Xr = A["X"][ia], R["X"][ir]

dX = np.linalg.norm(Xa - Xr, axis=2).max(axis=1)


def width(Xm):
    return np.hypot(Xm[..., half:, 0] - Xm[..., :half, 0],
                    Xm[..., half:, 1] - Xm[..., :half, 1]).mean(axis=1)


wa, wr = width(Xa), width(Xr)

# the applied electrophysiology (stiffness of the 78 cross-tube springs)
ka = A["k_elec"][ia]
kr = R["k_elec"][ir] if "k_elec" in R else None
if kr is not None:
    assert np.allclose(ka, kr, rtol=1e-12, atol=0.0), \
        "AFSI and reference applied different electro stiffnesses"

print(f"frames: AFSI {len(A['t'])}, IB2d {len(R['t'])}, compared {len(keep)} "
      f"(t = {t[0]:.4f} .. {t[-1]:.4f}, dt = {dt:g}, h = {h:g})")
print(f"{'t':>8} {'dXmax':>10} {'dXmax/h':>8} {'width A/I':>19} "
      f"{'k_elec AFT':>11}")
rows = []
for k in range(len(t)):
    rows.append((t[k], dX[k], dX[k] / h, wa[k], wr[k],
                 ka[k].mean(), ka[k].max()))
    if k % max(1, len(t) // 12) == 0 or k == len(t) - 1:
        print(f"{t[k]:8.4f} {dX[k]:10.3e} {dX[k] / h:8.2e} "
              f"{wa[k]:9.5f}/{wr[k]:9.5f} {ka[k].mean():11.4g}")
with open(os.path.join(out, "compare_table.csv"), "w") as fh:
    fh.write("t,dXmax,dXmax_over_h,width_afsi,width_ib2d,k_elec_mean,k_elec_max\n")
    for row in rows:
        fh.write(",".join(f"{v:.10g}" for v in row) + "\n")

print(f"\nmarker deviation : max {dX.max():.3e} m = {dX.max() / h:.2f} h, "
      f"median {np.median(dX):.2e} m = {np.median(dX) / h:.2e} h")
print(f"tube diameter    : AFSI {wa.min():.4f}..{wa.max():.4f}, IB2d "
      f"{wr.min():.4f}..{wr.max():.4f} (rest 1.0); max |dw| "
      f"{np.abs(wa - wr).max():.2e}")
if kr is not None:
    print(f"electro stiffness: applied (16v)^4 maxima AFSI/IB2d "
          f"{ka.max():.6g}/{kr.max():.6g}, identical to "
          f"{np.abs(ka - kr).max():.1e}")
if "u" in R and "u" in A:
    ua, ur = A["u"][ia[-1]], R["u"][ir[-1]]
    print(f"velocity field   : last common frame max|u| AFSI "
          f"{np.abs(ua).max():.4f} / IB2d {np.abs(ur).max():.4f}, "
          f"max|du| {np.abs(ua - ur).max():.4f}")

# --- 1. tube at several times -----------------------------------------------
nsel = 6
sel = np.linspace(0.0, t[-1], nsel)
fig, axes = plt.subplots(1, nsel, figsize=(2.4 * nsel, 4.4))
pad = 0.15
for ax, tsel in zip(np.atleast_1d(axes), sel):
    k = int(np.argmin(abs(t - tsel)))
    ax.plot(np.append(Xr[k, :half, 0], Xr[k, 0, 0]),
            np.append(Xr[k, :half, 1], Xr[k, 0, 1]), "k-", lw=1.8, alpha=0.55,
            label="IB2d")
    ax.plot(np.append(Xr[k, half:, 0], Xr[k, half, 0]),
            np.append(Xr[k, half:, 1], Xr[k, half, 1]), "k-", lw=1.8, alpha=0.55)
    ax.plot(np.append(Xa[k, :half, 0], Xa[k, 0, 0]),
            np.append(Xa[k, :half, 1], Xa[k, 0, 1]), "r--", lw=1.1, label="AFSI")
    ax.plot(np.append(Xa[k, half:, 0], Xa[k, half, 0]),
            np.append(Xa[k, half:, 1], Xa[k, half, 1]), "r--", lw=1.1)
    ax.set_title(f"t = {t[k]:.3f}")
    ax.set_aspect("equal")
    ax.set_xlim(0.7, 4.3)
    ax.set_ylim(1.7, 3.3)
np.atleast_1d(axes)[0].legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(out, "compare_shapes.png"), dpi=150)

# --- 2. histories -----------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
ax[0].semilogy(t[1:], dX[1:], "k-o", ms=3, label="markers")
ax[0].axhline(h, color="gray", ls=":", label="grid spacing h")
ax[0].set_title("marker deviation vs IB2d")
ax[0].set_xlabel("t")
ax[0].legend(fontsize=8)
ax[1].plot(t, wr, "k-", lw=2, label="IB2d")
ax[1].plot(t, wa, "r--", label="AFSI")
ax[1].set_title("cross-tube diameter")
ax[1].set_xlabel("t")
ax[1].legend(fontsize=8)
ax[2].plot(t, ka.mean(axis=1), "r--", label="AFSI")
if kr is not None:
    ax[2].plot(t, kr.mean(axis=1), "k-", lw=2, label="IB2d")
ax[2].set_title(r"applied electro stiffness $\overline{(16v)^4}$")
ax[2].set_xlabel("t")
ax[2].legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(out, "compare_history.png"), dpi=150)

print(f"figures written to {out}")
