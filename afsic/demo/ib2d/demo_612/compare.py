"""Compare an AFSI heart-tube *muscle* run with the pyIB2d reference.

    python compare.py [ib2d_reference.npz] [out_dir]

Inputs: ``plot/afsi_result.npz`` (main.py) and ``ib2d_reference.npz``
(run_reference.py).  Works for the muscle-driven members of the
``Example_HeartTube`` family (``EM_Pump_w_Calcium_and_Muscles``,
``Muscle_Model``, ``3_Element_Muscle_Model``): the structure
is a closed tube whose two walls are tied together by the example's muscle
bands, so the physical output is the **band length** (the tube's
cross-diameter) and its activation phase.

Frame alignment: the driver's pre-loop frame is the initial state (step 0) and
frame ``k >= 1`` is the state at step ``k * print_dump + 1`` (its VTK time stamp
is one step behind the state it wrote -- an upstream labelling quirk); main.py
dumps on exactly those steps.

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

sys.path.insert(0, here)
import ib2d_io  # noqa: E402

ex = ib2d_io.load_example(os.path.join(here, "ib2d_input"), index_base=1)
MUS = ex["muscles"]
MA, MB = MUS["conn"][:, 0], MUS["conn"][:, 1]
LFO, SK = MUS["LFO"], MUS["SK"]

R = np.load(f_ref)
A = np.load(f_afsi)
assert A["X"].shape[1] == R["X"].shape[1], "marker counts differ"
dt = float(R["dt"])
h = float(R["dx"])
half = A["X"].shape[1] // 2

steps_a = np.rint(A["t"] / dt).astype(int)
steps_r = np.rint(R["t"] / dt).astype(int)
keep = [k for k in range(min(len(steps_a), len(steps_r)))
        if steps_a[k] == steps_r[k] + (0 if k == 0 else 1)]
assert keep, f"no aligned frames: AFSI {steps_a} vs reference {steps_r}"
t = A["t"][np.array(keep)]
Xa, Xr = A["X"][keep], R["X"][keep]

dX = np.linalg.norm(Xa - Xr, axis=2).max(axis=1)


def band_length(Xm):
    """Length of every muscle band (the tube's local diameter)."""
    return np.hypot(Xm[..., MB, 0] - Xm[..., MA, 0],
                    Xm[..., MB, 1] - Xm[..., MA, 1])


la, lr = band_length(Xa), band_length(Xr)
wa, wr = la.mean(axis=1), lr.mean(axis=1)
# length-tension factor F1 = exp(-((L/LFO-1)/SK)^2) of the two runs
f1a = np.exp(-(((la / LFO) - 1.0) / SK) ** 2)
f1r = np.exp(-(((lr / LFO) - 1.0) / SK) ** 2)

print(f"frames: AFSI {len(A['t'])}, IB2d {len(R['t'])}, compared {len(keep)} "
      f"(t = {t[0]:.4f} .. {t[-1]:.4f}, dt = {dt:g}, h = {h:g}, "
      f"{len(MA)} muscle bands)")
print(f"{'t':>8} {'dXmax':>10} {'dXmax/h':>8} {'bandlen A/I':>19} "
      f"{'F1 A/I':>15}")
rows = []
for k in range(len(t)):
    rows.append((t[k], dX[k], dX[k] / h, wa[k], wr[k],
                 f1a[k].mean(), f1r[k].mean()))
    if k % max(1, len(t) // 12) == 0 or k == len(t) - 1:
        print(f"{t[k]:8.4f} {dX[k]:10.3e} {dX[k] / h:8.2e} "
              f"{wa[k]:9.5f}/{wr[k]:9.5f} "
              f"{f1a[k].mean():7.4f}/{f1r[k].mean():7.4f}")
with open(os.path.join(out, "compare_table.csv"), "w") as fh:
    fh.write("t,dXmax,dXmax_over_h,bandlen_afsi,bandlen_ib2d,"
             "F1_afsi,F1_ib2d\n")
    for row in rows:
        fh.write(",".join(f"{v:.10g}" for v in row) + "\n")

print(f"\nmarker deviation : max {dX.max():.3e} m = {dX.max() / h:.2f} h, "
      f"median {np.median(dX):.2e} m = {np.median(dX) / h:.2e} h")
print(f"tube diameter    : AFSI {wa.min():.4f}..{wa.max():.4f}, IB2d "
      f"{wr.min():.4f}..{wr.max():.4f} (rest 1.0); max |dw| "
      f"{np.abs(wa - wr).max():.2e}")
print(f"activation phase : AFSI min diameter at t={t[int(np.argmin(wa))]:.4f}, "
      f"IB2d at t={t[int(np.argmin(wr))]:.4f}")
print(f"length-tension   : mean F1 AFSI {f1a.mean():.5f}, IB2d "
      f"{f1r.mean():.5f}; max |dF1| {np.abs(f1a - f1r).max():.3e}")
if "u" in R and "u" in A:
    ua, ur = A["u"][keep[-1]], R["u"][keep[-1]]
    print(f"velocity field   : last common frame max|u| AFSI "
          f"{np.abs(ua).max():.4f} / IB2d {np.abs(ur).max():.4f}, "
          f"max|du| {np.abs(ua - ur).max():.4f}")

# --- 1. tube at several times -----------------------------------------------
nsel = 6
sel = np.linspace(0.0, t[-1], nsel)
fig, axes = plt.subplots(1, nsel, figsize=(2.4 * nsel, 4.4))
for ax, tsel in zip(np.atleast_1d(axes), sel):
    k = int(np.argmin(abs(t - tsel)))
    for row, (D, col) in enumerate(((Xr, "k-"), (Xa, "r--"))):
        Xm = D[k]
        ax.plot(np.append(Xm[:half, 0], Xm[0, 0]),
                np.append(Xm[:half, 1], Xm[0, 1]), col, lw=1.8, alpha=0.6,
                label="IB2d" if row == 0 else "AFSI")
        ax.plot(np.append(Xm[half:, 0], Xm[half, 0]),
                np.append(Xm[half:, 1], Xm[half, 1]), col, lw=1.8, alpha=0.6)
    ax.set_title(f"t = {t[k]:.3f}")
    ax.set_aspect("equal")
np.atleast_1d(axes)[0].legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(out, "compare_shapes.png"), dpi=150)

# --- 2. histories -----------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
ax[0].semilogy(t[1:], dX[1:], "k-o", ms=3, label="markers")
ax[0].axhline(h, color="gray", ls=":", label="grid spacing h")
ax[0].set_title("deviation vs IB2d")
ax[0].set_xlabel("t")
ax[0].legend(fontsize=8)
ax[1].plot(t, wr, "k-", lw=2, label="IB2d")
ax[1].plot(t, wa, "r--", label="AFSI")
ax[1].set_title("mean muscle-band length (tube diameter)")
ax[1].set_xlabel("t")
ax[1].legend(fontsize=8)
for k in range(0, len(t), max(1, len(t) // 8)):
    ax[2].plot(np.arange(len(MA)), lr[k], "k-", lw=1.0,
               alpha=0.3 + 0.7 * k / max(1, len(t) - 1))
    ax[2].plot(np.arange(len(MA)), la[k], "r--", lw=0.8,
               alpha=0.3 + 0.7 * k / max(1, len(t) - 1))
ax[2].set_title("band-length profiles (IB2d solid / AFSI dashed)")
ax[2].set_xlabel("muscle band index")
fig.tight_layout()
fig.savefig(os.path.join(out, "compare_history.png"), dpi=150)

print(f"figures written to {out}")
