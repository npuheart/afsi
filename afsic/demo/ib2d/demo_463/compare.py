"""Compare the AFSI porous/non-porous rubberbands with the pyIB2d reference.

    python compare.py [ib2d_reference.npz] [out_dir]

Inputs: ``plot/afsi_result_g*.npz`` (main.py) and ``ib2d_reference.npz``
(run_reference.py).  Two rings of 128 markers: the one on the left is flagged as
porous (fluid passes through it, so it is far more compliant), the one on the
right is impermeable.  The primary quantities are the two enclosed areas and
the marker-wise deviation to the reference.

Frame alignment: the pyIB2d driver dumps when its 0-based step counter ``cter``
is a positive multiple of ``print_dump``, i.e. at the *end* of step
``cter + 1``, and stamps the frame with ``cter * dt`` (one step behind the state
it writes); ``main.py`` dumps on exactly those steps, so frame ``k`` of the two
codes is the same step of the simulation.

Writes ``figures/compare_shapes.png``, ``figures/compare_history.png`` (areas,
deviation), ``figures/compare_fields.png`` (vorticity) and
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

f_afsi = os.path.join(here, "plot", f"afsi_result_g{os.environ.get('GRAD_DIV', '2.5')}.npz")
for alt in ("afsi_result_g2.5.npz", "afsi_result_g100.npz", "afsi_result.npz"):
    if not os.path.exists(f_afsi):
        f_afsi = os.path.join(here, "plot", alt)

R = np.load(f_ref)
A = np.load(f_afsi)
assert A["X"].shape[1] == R["X"].shape[1], "marker counts differ"

# the two files dump on the same simulation steps (frame k of each is the end
# of step k * print_dump + 1), so they are compared frame by frame
n = min(len(A["t"]), len(R["t"]))
assert n >= 2, "no common output times"
t = A["t"][:n]
Xa, Xr = A["X"][:n], R["X"][:n]
N_RING = 128
ring1, ring2 = np.arange(N_RING), np.arange(N_RING, 2 * N_RING)


def area(X):
    return 0.5 * abs(np.dot(X[:, 0], np.roll(X[:, 1], -1))
                     - np.dot(X[:, 1], np.roll(X[:, 0], -1)))


dX = np.array([np.linalg.norm(Xa[k] - Xr[k], axis=1).max() for k in range(n)])
dX_por = np.array([np.linalg.norm(Xa[k, ring1] - Xr[k, ring1], axis=1).max()
                   for k in range(n)])
dX_pln = np.array([np.linalg.norm(Xa[k, ring2] - Xr[k, ring2], axis=1).max()
                   for k in range(n)])
area_a = np.array([area(X[ring1]) for X in Xa])
area_r = np.array([area(X[ring1]) for X in Xr])
area2_a = np.array([area(X[ring2]) for X in Xa])
area2_r = np.array([area(X[ring2]) for X in Xr])

print(f"{'t':>7} {'dX porous':>10} {'dX plain':>10} {'A_por A/I':>19} "
      f"{'A_pln A/I':>19}")
rows = []
for k, tt in enumerate(t):
    rows.append((tt, dX_por[k], dX_pln[k], area_a[k], area_r[k],
                 area2_a[k], area2_r[k]))
    if k % max(1, len(t) // 12) == 0 or k == len(t) - 1:
        print(f"{tt:7.4f} {dX_por[k]:10.3e} {dX_pln[k]:10.3e} "
              f"{area_a[k]:9.5f}/{area_r[k]:9.5f} "
              f"{area2_a[k]:9.5f}/{area2_r[k]:9.5f}")
with open(os.path.join(out, "compare_table.csv"), "w") as fh:
    fh.write("t,dX_porous,dX_plain,area_porous_afsi,area_porous_ib2d,"
             "area_plain_afsi,area_plain_ib2d\n")
    for row in rows:
        fh.write(",".join(f"{v:.10g}" for v in row) + "\n")

print(f"\nmarker deviation : porous ring max {dX_por.max():.3e} m "
      f"({dX_por.max() / (2 / 64):.2e} h), plain ring max "
      f"{dX_pln.max():.3e} m ({dX_pln.max() / (2 / 64):.2e} h) "
      f"(h = Lx/Nx = {2 / 64:g})")
print(f"porous ring area : AFSI {area_a[0]:.6f} -> {area_a[-1]:.6f}, IB2d "
      f"{area_r[0]:.6f} -> {area_r[-1]:.6f}, max|dA| {np.abs(area_a - area_r).max():.2e}")
print(f"plain ring area  : AFSI {area2_a[0]:.6f} -> {area2_a[-1]:.6f}, IB2d "
      f"{area2_r[0]:.6f} -> {area2_r[-1]:.6f}, max|dA| "
      f"{np.abs(area2_a - area2_r).max():.2e}")

# --- 1. ring at several times -------------------------------------------------
nsel = 6
sel = np.linspace(0.0, t[-1], nsel)
fig, axes = plt.subplots(1, nsel, figsize=(2.4 * nsel, 3.2))
xlim = (Xr[..., 0].min() - 0.05, Xr[..., 0].max() + 0.05)
ylim = (Xr[..., 1].min() - 0.05, Xr[..., 1].max() + 0.05)
for ax, tsel in zip(np.atleast_1d(axes), sel):
    k = int(np.argmin(abs(t - tsel)))
    for rr in (ring1, ring2):
        for X, sty, lab in ((Xr, "k-", "IB2d"), (Xa, "r--", "AFSI")):
            ax.plot(np.append(X[k, rr, 0], X[k, rr[0], 0]),
                    np.append(X[k, rr, 1], X[k, rr[0], 1]), sty, lw=1.6,
                    alpha=0.6 if lab == "IB2d" else 1.0,
                    label=lab if rr is ring1 else None)
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
ax[2].semilogy(t[1:], dX_por[1:], "k-o", ms=3, label="porous ring")
ax[2].semilogy(t[1:], dX_pln[1:], "b-s", ms=3, label="plain ring")
ax[2].axhline(R["dx"], color="gray", ls=":", label="grid spacing h")
ax[2].set_title("marker deviation")
ax[2].set_xlabel("t")
ax[2].legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(out, "compare_history.png"), dpi=150)

# --- 3. vorticity at the last common time --------------------------------------
if "u" in R and "u" in A:
    k = len(t) - 1
    ua, ur = A["u"][k], R["u"][k]
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
