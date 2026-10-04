"""Compare the AFSI empty-channel parabolic flow with the pyIB2d reference.

    python compare.py [ib2d_reference.npz] [out_dir]

Inputs: ``plot/afsi_result.npz`` (main.py) and ``ib2d_reference.npz``
(run_reference.py).  Both codes evaluate the *same* ported
``please_Compute_External_Forcing.py``, so the comparison isolates the solver
and the immersed-boundary machinery.

Frame alignment: the pyIB2d driver dumps when its 0-based step counter
``cter`` is a positive multiple of ``print_dump`` -- i.e. at the end of step
``cter + 1`` -- and stamps that frame with ``cter * dt``, one step behind the
state it writes (upstream labelling quirk).  ``main.py`` dumps on exactly those
steps and stamps the true time, so AFSI frame ``k`` and reference frame ``k``
are the same step whenever ``round(t_afsi/dt) == round(t_ref/dt) + 1``; that is
checked here and only the matching frames are used.

Errors are reported inside and outside the four-node delta-function band of the
Lagrangian markers (derived from the target points); the target profile and the
force window are read back from the ported module itself, and the module's force
is checked against AFSI's force field evaluated at AFSI's own velocity.

The shipped ``dt`` puts the velocity-target penalty at the marginal gain
``k dt / rho = 1``, and IB2d's collocated FFT scheme then grows a grid-scale
spike inside the wall band; with ``dt`` halved (gain 0.5) the spike disappears
and the two solutions agree to a couple of percent.  Every
``ib2d_reference<TAG>.npz`` -- ``plot<TAG>/afsi_result.npz`` pair found next to
this file is compared automatically, so the ``--dt 5e-5`` cross-check is picked
up without arguments.

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
f_ref = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "ib2d_reference.npz")
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "figures")
os.makedirs(out, exist_ok=True)

f_afsi = os.path.join(here, "plot", "afsi_result.npz")

sys.path.insert(0, here)
import ib2d_io  # noqa: E402

sys.path.insert(0, os.path.join(here, "ib2d_input"))
import please_Compute_External_Forcing as mod  # noqa: E402

P = ib2d_io.read_input2d(os.path.join(here, "ib2d_input", "input2d"))
Lx = float(P["Lx"])
Ly = float(P["Ly"])

class Grid:
    """Node grid of one (AFSI, reference) pair, plus the module-derived masks."""

    def __init__(self, nx, ny, dt, anchors, nb):
        self.nx, self.ny = nx, ny
        self.hx, self.hy = Lx / nx, Ly / ny
        self.dt = dt
        self.XNODES = np.arange(nx) * self.hx
        self.YNODES = np.arange(ny) * self.hy
        # IB2d's ds and grid_Info, as the driver builds them
        self.GRID_INFO = [nx, ny, Lx, Ly, self.hx, self.hy, int(P["supp"]), nb,
                          Lx / (2 * nx)]
        # the module's own force window, evaluated on this grid
        self.win = np.zeros((ny, nx), bool)
        i, j = mod._indices_to_apply_force(self.XNODES, self.YNODES).T
        self.win[j, i] = True
        # four-point delta support of the markers: the IB band
        self.band = np.zeros((ny, nx), bool)
        for x0, y0 in anchors:
            i0 = int(round(x0 / self.hx))
            j0 = int(round(y0 / self.hy))
            self.band[max(0, j0 - 2):j0 + 3, max(0, i0 - 2):i0 + 3] = True
        self.away = ~self.band
        self.core = self.win & self.away
        # rows of the forcing window away from the band: the channel section
        # the example drives (outside of it only weak return flow)
        self.col = self.win.any(axis=1) & ~self.band[:, nx // 2]

    def module_force(self, t, ux, uy):
        """(Fx, Fy) of the ported arbitrary-external-force module."""
        return mod.please_Compute_External_Forcing(
            self.dt, t, self.XNODES, self.YNODES, self.GRID_INFO, ux.copy(),
            uy.copy(), 1, None)[:2]

    def target_velocity(self, t):
        """u^tar of the module, read back through Fx = -k (u - u^tar) at u=0."""
        fx, fy = self.module_force(t, np.zeros((self.ny, self.nx)),
                                   np.zeros((self.ny, self.nx)))
        k = getattr(mod, "K_STIFF", 1.0)
        return -fx / k, -fy / k




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
        ua, ur = A["u"][k], R["u"][k]
        du = np.linalg.norm(ua - ur, axis=-1)
        t_k = A["t"][k]
        # the ported module must add exactly its force to the AFSI field;
        # nodes carrying a target force as well are excluded via ``core``
        f_mod = G.module_force(t_k, ua[..., 0], ua[..., 1])[0]
        res = np.abs(A["f"][k][..., 0] - f_mod)[G.core]
        df = np.abs(A["f"][k] - R["f"][k])
        rel = (np.linalg.norm(du[G.away]) / np.linalg.norm(ur[G.away])
               if np.linalg.norm(ur[G.away]) else 0.0)
        rows.append((t_k,
                     np.linalg.norm(A["X"][k] - R["X"][k], axis=1).max(),
                     du[G.away].max(), du[G.band].max(), rel,
                     np.abs(ua).max(), np.abs(ur).max(),
                     np.abs(ua[G.win]).max(), np.abs(ur[G.win]).max(),
                     ur[G.col, G.nx // 2, 0].sum() * G.hy,
                     ua[G.col, G.nx // 2, 0].sum() * G.hy,
                     df[G.away].max(), res.max() if res.size else 0.0))
    return keep, rows


def report(tag, G, A, R, keep, rows):
    print(f"\n=== {tag} (dt = {G.dt:g}, h = {G.hx:g}, {len(keep)} aligned "
          f"frames, t = {rows[0][0]:.5f} .. {rows[-1][0]:.5f})")
    print(f"{'t':>8} {'dXmax/h':>9} {'max|du|away':>12} {'relL2_away':>11} "
          f"{'max|du|band':>12} {'max|u| a/r':>19} {'window a/r':>19} "
          f"{'flux a/r':>19} {'df away':>9}")
    for r in rows:
        print(f"{r[0]:8.5f} {r[1] / G.hx:9.2e} {r[2]:12.3e} {r[4]:11.3e} "
              f"{r[3]:12.3e} {r[5]:9.4f}/{r[6]:<9.4f} "
              f"{r[7]:9.4f}/{r[8]:<9.4f} {r[10]:9.5f}/{r[9]:<9.5f} "
              f"{r[11]:9.3e}")
    r = rows[-1]
    print(f"markers          : max|dX| {r[1]:.3e} m = {r[1] / G.hx:.2e} h")
    print(f"velocity, away from the IB band: max|du| {r[2]:.3e} = "
          f"{r[2] / r[6]:.2e} max|u|, relative L2 error {r[4]:.3e} "
          f"({int(G.away.sum())} of {G.away.size} nodes)")
    print(f"velocity, inside the IB band  : max|du| {r[3]:.3e} = "
          f"{r[3] / r[6]:.2f} max|u|")
    print(f"forcing window ({int(G.win.sum())} nodes): max|u| AFSI {r[7]:.4f} / IB2d "
          f"{r[8]:.4f} ({abs(r[7] - r[8]) / r[8]:.2e} relative)")
    print(f"channel flux, middle column over the window rows "
          f"({int(G.col.sum())} rows): AFSI {r[10]:.5f}, IB2d {r[9]:.5f} "
          f"({abs(r[10] - r[9]) / abs(r[9]):.2e} relative)")
    print(f"force field      : max|df| {r[11]:.3e} away from the band, "
          f"|f|max {np.abs(R['f'][keep[-1]]).max():.3e}")
    if len(rows) > 3:
        sel = np.unique(np.linspace(1, len(rows) - 1, 4).astype(int))
        print("window max|u| AFSI/IB2d at t = "
              + ", ".join(f"{rows[i][0]:.4f}: {rows[i][7]:.4f}/{rows[i][8]:.4f}"
                          for i in sel))
    print(f"forcing check    : |f_x - F_x^module(u^AFSI)| <= {r[12]:.3e} on the "
          f"window nodes away from the band")


def load_pair(ref_path, afsi_path):
    """Grid + data of one pair (the AFSI file carries dt, nx, ny)."""
    A = np.load(afsi_path)
    R = np.load(ref_path)
    dt = float(A["dt"]) if "dt" in A.files else float(
        R["dt"] if "dt" in R.files else P["dt"])
    ny, nx = A["u"].shape[1:3]
    G = Grid(nx, ny, dt, A["Xa"][0], A["X"].shape[1])
    return G, A, R


G, A, R = load_pair(f_ref, f_afsi)
keep, rows = metrics(G, A, R)
report("shipped setup", G, A, R, keep, rows)
main_pair = (G, A, R, keep, rows)

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
Xa, Xr, Xan = A["X"][keep], R["X"][keep], A["Xa"][keep]
dX = np.array([r[1] for r in rows])

with open(os.path.join(out, "compare_table.csv"), "w") as fh:
    fh.write("t,dXmax,du_max_away,du_max_band,relL2_away,umax_afsi,umax_ib2d,"
             "umax_window_afsi,umax_window_ib2d,flux_ib2d,flux_afsi,df_max,"
             "force_residual\n")
    for r in rows:
        fh.write(",".join(f"{v:.10g}" for v in r) + "\n")

t = A["t"][keep]
Xa, Xr = A["X"][keep], R["X"][keep]
dX = np.array([r[1] for r in rows])

with open(os.path.join(out, "compare_table.csv"), "w") as fh:
    fh.write("t,dXmax,du_max_away,du_max_band,relL2_away,umax_afsi,umax_ib2d,"
             "umax_window_afsi,umax_window_ib2d,flux_ib2d,flux_afsi,df_max,"
             "force_residual\n")
    for r in rows:
        fh.write(",".join(f"{v:.10g}" for v in r) + "\n")

# --- 1. velocity profiles ---------------------------------------------------
ksel = [keep[0], keep[len(keep) // 2], keep[-1]] if len(keep) >= 3 else keep
i_w = int(np.argmax(G.win.sum(axis=0)))        # busiest forcing column
fig, axes = plt.subplots(2, len(ksel), figsize=(4.0 * len(ksel), 6.4),
                         sharey=True)
for c, k in enumerate(ksel):
    tar = G.target_velocity(A["t"][k])[0]
    for row, i in enumerate((i_w, G.nx // 2)):
        ax = axes[row, c]
        ax.plot(R["u"][k][:, i, 0], G.YNODES, "k-", lw=2.0, alpha=0.6,
                label="IB2d")
        ax.plot(A["u"][k][:, i, 0], G.YNODES, "r--", lw=1.2, label="AFSI")
        if row == 0 and G.win[:, i].any():
            m = G.win[:, i]
            ax.plot(tar[m, i], G.YNODES[m], "b:", lw=1.0, label="target")
        if G.band[:, i].any():
            ax.plot(R["u"][k][G.band[:, i], i, 0], G.YNODES[G.band[:, i]],
                    "k.", ms=2)
        ax.set_xlabel("$u_x$")
        ax.set_title(f"x = {G.XNODES[i]:.4f}, t = {A['t'][k]:.5f}")
        if row == 0 and c == 0:
            ax.legend(fontsize=8)
axes[0, 0].set_ylabel("y")
axes[1, 0].set_ylabel("y")
fig.tight_layout()
fig.savefig(os.path.join(out, "compare_profiles.png"), dpi=150)

# --- 2. histories -----------------------------------------------------------
fig, ax = plt.subplots(1, 4, figsize=(17, 3.6))
ax[0].semilogy(t, dX, "k-o", ms=3)
ax[0].axhline(G.hx, color="gray", ls=":", label="h")
ax[0].set_title("marker deviation from IB2d")
ax[0].set_xlabel("t")
ax[0].legend(fontsize=8)
ax[1].plot(t, [r_[6] for r_ in rows], "k-", lw=2, label="IB2d")
ax[1].plot(t, [r_[5] for r_ in rows], "r--", label="AFSI")
ax[1].set_title(r"$\max|u|$")
ax[1].set_xlabel("t")
ax[1].legend(fontsize=8)
ax[2].plot(t, [r_[9] for r_ in rows], "k-", lw=2, label="IB2d")
ax[2].plot(t, [r_[10] for r_ in rows], "r--", label="AFSI")
ax[2].set_title(r"flux $\int u_x\,dy$ through the middle column")
ax[2].set_xlabel("t")
ax[2].legend(fontsize=8)
ax[3].semilogy(t, [max(r_[2], 1e-16) for r_ in rows], "b-s", ms=3,
               label="away from the IB band")
ax[3].semilogy(t, [r_[3] for r_ in rows], "k-o", ms=3, label="IB band")
ax[3].set_title(r"$|u^{AFSI}-u^{IB2d}|$")
ax[3].set_xlabel("t")
ax[3].legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(out, "compare_history.png"), dpi=150)

# --- 3. fields at the last aligned frame ------------------------------------
k = keep[-1]
ua, ur = A["u"][k], R["u"][k]
fig, ax = plt.subplots(1, 4, figsize=(19, 4.4))
vmax = max(np.abs(ur[..., 0]).max(), np.abs(ua[..., 0]).max())
for a_, u, title in ((ax[0], ur, "IB2d"), (ax[1], ua, "AFSI")):
    c = a_.pcolormesh(G.XNODES, G.YNODES, u[..., 0], shading="auto",
                      cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    a_.plot(R["X"][k][:, 0], R["X"][k][:, 1], "k.", ms=0.6)
    a_.set_title(f"{title}: $u_x$, t = {t[-1]:.5f}")
    fig.colorbar(c, ax=a_, shrink=0.85)
dv = ua[..., 0] - ur[..., 0]
lim = np.percentile(np.abs(dv), 99.5)
c = ax[2].pcolormesh(G.XNODES, G.YNODES, dv, shading="auto", cmap="RdBu_r",
                     vmin=-lim, vmax=lim)
ax[2].set_title("$u_x^{AFSI}-u_x^{IB2d}$")
fig.colorbar(c, ax=ax[2], shrink=0.85)
du = np.linalg.norm(ua - ur, axis=-1)
c = ax[3].pcolormesh(G.XNODES, G.YNODES, du, shading="auto", cmap="viridis",
                     vmin=0, vmax=du[G.away].max())
ax[3].contour(G.XNODES, G.YNODES, G.band.astype(float), levels=[0.5],
              colors="w", lw=0.6)
ax[3].set_title(f"$|\\Delta u|$, max {du[G.away].max():.2e} away from the band")
fig.colorbar(c, ax=ax[3], shrink=0.85)
fig.tight_layout()
fig.savefig(os.path.join(out, "compare_fields.png"), dpi=150)

print(f"figures written to {out}")
