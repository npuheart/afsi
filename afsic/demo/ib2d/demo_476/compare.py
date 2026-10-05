"""Compare AFSI's pure-diffusion passive scalar with the pyIB2d reference.

    python compare.py [ib2d_reference.npz] [out_dir]

Inputs: ``plot/afsi_result.npz`` (main.py) and ``ib2d_reference.npz``
(run_reference.py).

Frame alignment: the pyIB2d driver dumps when its 0-based step counter
``cter`` is a positive multiple of ``print_dump`` -- i.e. at the end of step
``cter + 1`` -- and stamps that frame with ``cter * dt``, one step behind the
state it writes (upstream labelling quirk).  ``main.py`` dumps on exactly those
steps and stamps the true time, so AFSI frame ``k`` and reference frame ``k``
are the same step whenever ``round(t_afsi/dt) == round(t_ref/dt) + 1``; that is
checked here and only the matching frames are used.

Because both codes advance the *same* explicit upwind/central
advection-diffusion operator on the same collocated grid, the concentration
fields are expected to agree to round-off; the reported errors are therefore a
direct check of the ported operator (and of the AFSI velocity field, which must
stay identically zero: the chamber is pinned and nothing drives the fluid).

Writes ``figures/compare_profiles.png``, ``figures/compare_history.png``,
``figures/compare_fields.png`` and ``figures/compare_table.csv``.
"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

here = os.path.dirname(os.path.abspath(__file__))
f_ref = (sys.argv[1] if len(sys.argv) > 1
         else os.path.join(here, "ib2d_reference.npz"))
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "figures")
os.makedirs(out, exist_ok=True)

f_afsi = os.path.join(here, "plot", "afsi_result.npz")

sys.path.insert(0, here)
import ib2d_io  # noqa: E402

P = ib2d_io.read_input2d(os.path.join(here, "ib2d_input", "input2d"))
Lx, Ly = float(P["Lx"]), float(P["Ly"])


class Grid:
    """Node grid of one (AFSI, reference) pair."""

    def __init__(self, nx, ny, dt, anchors):
        self.nx, self.ny = nx, ny
        self.hx, self.hy = Lx / nx, Ly / ny
        self.dt = dt
        self.XNODES = np.arange(nx) * self.hx
        self.YNODES = np.arange(ny) * self.hy
        # four-point delta support of the markers (the chamber walls)
        self.band = np.zeros((ny, nx), bool)
        for x0, y0 in anchors:
            i0 = int(round(x0 / self.hx))
            j0 = int(round(y0 / self.hy))
            self.band[max(0, j0 - 2):j0 + 3, max(0, i0 - 2):i0 + 3] = True
        self.away = ~self.band


def metrics(G, A, R):
    """Per-frame comparison of one (AFSI, reference) pair on grid ``G``."""
    dt = G.dt
    steps_a = np.rint(A["t"] / dt).astype(int)
    steps_r = np.rint(R["t"] / dt).astype(int)
    # the driver's pre-loop dump is the initial state (step 0); every later
    # frame k is the state at step steps_r[k] + 1
    keep = [k for k in range(min(len(steps_a), len(steps_r)))
            if steps_a[k] == steps_r[k] + (0 if k == 0 else 1)]
    assert keep, f"no aligned frames: AFSI {steps_a} vs reference {steps_r}"

    rows = []
    for k in keep:
        du = np.linalg.norm(A["u"][k] - R["u"][k], axis=-1)
        Ca, Cr = A["C"][k], R["C"][k]
        dC = np.abs(Ca - Cr)
        ref = np.linalg.norm(Cr)
        rows.append((A["t"][k],
                     np.linalg.norm(A["X"][k] - R["X"][k], axis=1).max(),
                     np.abs(A["u"][k]).max(), np.abs(R["u"][k]).max(),
                     du.max(),
                     dC.max(), np.linalg.norm(Ca - Cr) / ref if ref else 0.0,
                     Ca.max(), Cr.max(),
                     Ca.sum() * G.hx * G.hy, Cr.sum() * G.hx * G.hy,
                     dC[G.away].max(), dC[G.band].max()))
    return keep, rows


def report(tag, G, A, R, keep, rows):
    print(f"\n=== {tag} (dt = {G.dt:g}, h = {G.hx:g}, {len(keep)} aligned "
          f"frames, t = {rows[0][0]:.5f} .. {rows[-1][0]:.5f})")
    print(f"{'t':>8} {'dXmax/h':>10} {'max|u| a/r':>19} {'max|dC|':>11} "
          f"{'relL2(C)':>10} {'maxC a/r':>19} {'sum C a/r':>21} "
          f"{'dC away':>10} {'dC band':>10}")
    for r in rows:
        print(f"{r[0]:8.5f} {r[1] / G.hx:10.2e} {r[2]:9.2e}/{r[3]:<9.2e} "
              f"{r[5]:11.3e} {r[6]:10.3e} {r[7]:9.5f}/{r[8]:<9.5f} "
              f"{r[9]:10.6f}/{r[10]:<10.6f} {r[11]:10.2e} {r[12]:10.2e}")
    r = rows[-1]
    print(f"markers          : max|dX| {r[1]:.3e} m = {r[1] / G.hx:.2e} h "
          f"(pinned chamber)")
    print(f"velocity         : max|u| AFSI {r[2]:.3e} / IB2d {r[3]:.3e} "
          f"(max|du| {r[4]:.3e}) -- the flow stays identically zero")
    print(f"concentration    : max|dC| {r[5]:.3e}, relative L2 error {r[6]:.3e}")
    print(f"                   peak  AFSI {r[7]:.6f} / IB2d {r[8]:.6f} "
          f"({abs(r[7] - r[8]) / r[8]:.2e} relative)")
    print(f"                   total AFSI {r[9]:.8f} / IB2d {r[10]:.8f} "
          f"({abs(r[9] - r[10]) / r[10]:.2e} relative)")
    print(f"                   max|dC| {r[11]:.3e} away from the wall band, "
          f"{r[12]:.3e} inside it")


def load_pair(ref_path, afsi_path):
    """Grid + data of one pair (the AFSI file carries dt, nx, ny)."""
    A = np.load(afsi_path)
    R = np.load(ref_path)
    dt = float(A["dt"]) if "dt" in A.files else float(
        R["dt"] if "dt" in R.files else P["dt"])
    ny, nx = A["C"].shape[1:3]
    G = Grid(nx, ny, dt, A["Xa"][0])
    return G, A, R


G, A, R = load_pair(f_ref, f_afsi)
keep, rows = metrics(G, A, R)
report("shipped setup", G, A, R, keep, rows)

# cross-checks: ``ib2d_reference<TAG>.npz`` pairs with ``plot<TAG>/``
import glob  # noqa: E402

for ref_path in sorted(glob.glob(os.path.join(here, "ib2d_reference*.npz"))):
    tag = os.path.basename(ref_path)[len("ib2d_reference"):-len(".npz")]
    if not tag:
        continue
    afsi_path = os.path.join(here, f"plot{tag}", "afsi_result.npz")
    if not os.path.exists(afsi_path):
        continue
    Gx, Ax, Rx = load_pair(ref_path, afsi_path)
    kx, rowsx = metrics(Gx, Ax, Rx)
    report(f"cross-check {tag}", Gx, Ax, Rx, kx, rowsx)

t = A["t"][keep]
dX = np.array([r[1] for r in rows])

with open(os.path.join(out, "compare_table.csv"), "w") as fh:
    fh.write("t,dXmax,umax_afsi,umax_ib2d,du_max,dC_max,relL2_C,Cmax_afsi,"
             "Cmax_ib2d,sumC_afsi,sumC_ib2d,dC_max_away,dC_max_band\n")
    for r in rows:
        fh.write(",".join(f"{v:.10g}" for v in r) + "\n")

# --- 1. concentration profiles along the two centre lines -------------------
ksel = [keep[0], keep[len(keep) // 2], keep[-1]] if len(keep) >= 3 else keep
fig, axes = plt.subplots(2, len(ksel), figsize=(4.0 * len(ksel), 6.2),
                         sharey=True)
for c, k in enumerate(ksel):
    for row, label in enumerate(("x", "y")):
        ax = axes[row, c]
        if row == 0:                     # along x at y = 0.5
            Ca, Cr = A["C"][k][G.ny // 2, :], R["C"][k][G.ny // 2, :]
            coord = G.XNODES
        else:                            # along y at x = 0.5
            Ca, Cr = A["C"][k][:, G.nx // 2], R["C"][k][:, G.nx // 2]
            coord = G.YNODES
        ax.plot(coord, Cr, "k-", lw=2.0, alpha=0.6, label="IB2d")
        ax.plot(coord, Ca, "r--", lw=1.2, label="AFSI")
        ax.set_xlabel(label)
        ax.set_title(f"t = {A['t'][k]:.5f}")
        if row == 0 and c == 0:
            ax.legend(fontsize=8)
axes[0, 0].set_ylabel("C (y = 0.5)")
axes[1, 0].set_ylabel("C (x = 0.5)")
fig.tight_layout()
fig.savefig(os.path.join(out, "compare_profiles.png"), dpi=150)

# --- 2. histories -----------------------------------------------------------
fig, ax = plt.subplots(1, 4, figsize=(17, 3.6))
ax[0].plot(t, [r[8] for r in rows], "k-", lw=2, label="IB2d")
ax[0].plot(t, [r[7] for r in rows], "r--", label="AFSI")
ax[0].set_title(r"peak concentration $\max C$")
ax[0].set_xlabel("t")
ax[0].legend(fontsize=8)
ax[1].plot(t, [r[10] for r in rows], "k-", lw=2, label="IB2d")
ax[1].plot(t, [r[9] for r in rows], "r--", label="AFSI")
ax[1].set_title(r"total $\sum C\,h^2$ (conserved)")
ax[1].set_xlabel("t")
ax[1].legend(fontsize=8)
ax[2].semilogy(t, [max(r[5], 1e-18) for r in rows], "b-o", ms=3, label="max")
ax[2].semilogy(t, [max(r[6], 1e-18) for r in rows], "g-s", ms=3,
               label="relative L2")
ax[2].set_title("concentration error")
ax[2].set_xlabel("t")
ax[2].legend(fontsize=8)
ax[3].semilogy(t, [max(r[4], 1e-18) for r in rows], "b-s", ms=3)
ax[3].set_title(r"$\max|u^{AFSI} - u^{IB2d}|$")
ax[3].set_xlabel("t")
fig.tight_layout()
fig.savefig(os.path.join(out, "compare_history.png"), dpi=150)

# --- 3. fields at the last aligned frame ------------------------------------
k = keep[-1]
Ca, Cr = A["C"][k], R["C"][k]
dC = Ca - Cr
fig, ax = plt.subplots(1, 4, figsize=(19, 4.4))
vmax = max(Ca.max(), Cr.max())
for a_, C, title in ((ax[0], Cr, "IB2d"), (ax[1], Ca, "AFSI")):
    c = a_.pcolormesh(G.XNODES, G.YNODES, C, shading="auto", cmap="inferno",
                      vmin=0, vmax=vmax)
    a_.plot(R["X"][k][:, 0], R["X"][k][:, 1], "c.", ms=0.6)
    a_.set_title(f"{title}: C, t = {t[-1]:.5f}")
    fig.colorbar(c, ax=a_, shrink=0.85)
lim = max(np.abs(dC).max(), 1e-18)
c = ax[2].pcolormesh(G.XNODES, G.YNODES, dC, shading="auto", cmap="RdBu_r",
                     vmin=-lim, vmax=lim)
ax[2].set_title(f"$C^{{AFSI}} - C^{{IB2d}}$, max {lim:.2e}")
fig.colorbar(c, ax=ax[2], shrink=0.85)
c = ax[3].pcolormesh(G.XNODES, G.YNODES, np.abs(dC), shading="auto",
                     cmap="viridis", vmin=0, vmax=lim)
ax[3].set_title(r"$|\Delta C|$")
fig.colorbar(c, ax=ax[3], shrink=0.85)
fig.tight_layout()
fig.savefig(os.path.join(out, "compare_fields.png"), dpi=150)

print(f"figures written to {out}")
