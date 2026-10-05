"""Compare AFSI's massive (ghost) boundary points with the pyIB2d reference.

    python compare.py [ib2d_reference.npz] [out_dir]

Inputs: ``plot/afsi_result.npz`` (main.py) and ``ib2d_reference.npz``
(run_reference.py).

Frame alignment: the pyIB2d driver dumps when its 0-based step counter
``cter`` is a positive multiple of ``print_dump`` -- i.e. at the end of step
``cter + 1`` -- and stamps that frame with ``cter * dt``, one step behind the
state it writes (upstream labelling quirk).  ``main.py`` dumps on exactly those
steps and stamps the true time.

The headline metric is the downstream drift of the three cells' ghost masses
(pyIB2d never writes them to VTK; ``run_reference.py`` records them), so
``compare_trajectories.png`` shows the ghost centroids of the three mass groups
and their AFSI/IB2d deviation.

Writes ``figures/compare_trajectories.png``, ``figures/compare_history.png``,
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

f_afsi = os.path.join(os.environ.get("OUTPUT_PATH", os.path.join(here, "plot")),
                      "afsi_result.npz")

sys.path.insert(0, here)
import ib2d_io  # noqa: E402

P = ib2d_io.read_input2d(os.path.join(here, "ib2d_input", "input2d"))
Lx, Ly = float(P["Lx"]), float(P["Ly"])


def _min_image(d, Lx, Ly):
    """Fold a ghost-minus-marker offset into [-L/2, L/2] (IB2d's convention is
    that the ghost follows its marker across the periodic boundary)."""
    d = d.copy()
    for c, L in enumerate((Lx, Ly)):
        over = np.abs(d[:, c]) > L / 2
        d[over, c] -= np.sign(d[over, c]) * L
    return d


class Grid:
    """Node grid of one (AFSI, reference) pair."""

    def __init__(self, nx, ny, dt):
        self.nx, self.ny = nx, ny
        self.hx, self.hy = Lx / nx, Ly / ny
        self.dt = dt
        self.XNODES = np.arange(nx) * self.hx
        self.YNODES = np.arange(ny) * self.hy


def align(A, R):
    """Indices of the frames that describe the same step in both codes."""
    dt = float(A["dt"])
    steps_a = np.rint(A["t"] / dt).astype(int)
    steps_r = np.rint(R["t"] / dt).astype(int)
    # the driver's pre-loop dump is the initial state (step 0); every later
    # frame k is the state at step steps_r[k] + 1
    keep = [k for k in range(min(len(steps_a), len(steps_r)))
            if steps_a[k] == steps_r[k] + (0 if k == 0 else 1)]
    assert keep, f"no aligned frames: AFSI {steps_a} vs reference {steps_r}"
    return keep


def _mass_groups(R):
    """Unique ghost masses -> the ghost indices carrying them."""
    if "mass_M" not in R.files or R["mass_M"].size == 0:
        return []
    M = R["mass_M"]
    return [(m, np.where(M == m)[0]) for m in np.unique(M)]


def metrics(G, A, R, keep):
    has_m = "Xm" in A.files and A["Xm"].size and "Xm" in R.files
    rows = []
    for k in keep:
        Xa, Xr = A["X"][k], R["X"][k]
        ua, ur = np.abs(A["u"][k]).max(), np.abs(R["u"][k]).max()
        du = np.linalg.norm(A["u"][k] - R["u"][k], axis=-1).max()
        r = [A["t"][k], np.linalg.norm(Xa - Xr, axis=1).max(), ua, ur, du]
        if has_m:
            Ma, Mr = A["Xm"][k], R["Xm"][k]
            ids = R["mass_ids"]
            d = np.linalg.norm(Ma - Mr, axis=1)
            # the ghost only carries its spring offset, so compare that offset
            # (minimum-image, the markers wrap in the periodic box while the
            # free ghosts do not): this isolates the mass-spring coupling from
            # the markers' own fluid-coupling error
            off_a = _min_image(Ma - A["X"][k][ids], Lx, Ly)
            off_r = _min_image(Mr - R["X"][k][ids], Lx, Ly)
            r += [d.max(), np.sqrt((d ** 2).mean()),
                  np.linalg.norm(A["Vm"][k] - R["Vm"][k], axis=1).max(),
                  np.linalg.norm(off_a - off_r, axis=1).max(),
                  np.abs(Mr - R["Xm"][0]).max()]
        if "C" in A.files and A["C"].size and "C" in R.files:
            Ca, Cr = A["C"][k], R["C"][k]
            ref = np.linalg.norm(Cr)
            r += [np.abs(Ca - Cr).max(),
                  np.linalg.norm(Ca - Cr) / ref if ref else 0.0]
        rows.append(r)
    return rows


FMT = ("t,dXmax_markers,umax_afsi,umax_ib2d,du_max,"
       "dXm_max,dXm_rms,dVm_max,dOffset_max,mass_disp_ref,"
       "dC_max,relL2_C")
COLS = FMT.split(",")


def report(tag, G, A, R, keep, rows):
    has_m = "Xm" in A.files and A["Xm"].size
    has_c = "C" in A.files and A["C"].size
    print(f"\n=== {tag} (dt = {G.dt:g}, h = {G.hx:g}, {len(keep)} aligned "
          f"frames, t = {rows[0][0]:.5f} .. {rows[-1][0]:.5f})")
    print(f"{'t':>8} {'dXmax/h':>9} {'max|u| a/r':>18} {'max|du|':>9}"
          + (f" {'dXm_max/h':>10} {'dXm_rms/h':>10} {'dVm_max':>9}"
             f" {'dOff/h':>9} {'ghost disp':>11}" if has_m else "")
          + (f" {'dCmax':>10} {'relL2(C)':>9}" if has_c else ""))
    for r in rows:
        s = (f"{r[0]:8.5f} {r[1] / G.hx:9.2e} {r[2]:8.2e}/{r[3]:<8.2e} "
             f"{r[4]:9.2e}")
        if has_m:
            s += (f" {r[5] / G.hx:10.2e} {r[6] / G.hx:10.2e} {r[7]:9.2e} "
                  f"{r[8] / G.hx:9.2e} {r[9]:11.4f}")
        if has_c:
            s += f" {r[-2]:10.3e} {r[-1]:9.3e}"
        print(s)
    r = rows[-1]
    print(f"markers      : max|dX| {r[1]:.3e} m = {r[1] / G.hx:.2e} h")
    print(f"fluid        : max|u| AFSI {r[2]:.3e} / IB2d {r[3]:.3e} "
          f"({abs(r[2] - r[3]) / max(r[3], 1e-30):.2e} relative), "
          f"max|du| {r[4]:.3e}")
    if has_m:
        print(f"MASS POINTS  : max|dXm| {r[5]:.3e} m = {r[5] / G.hx:.2e} h, "
              f"rms {r[6]:.3e} m = {r[6] / G.hx:.2e} h  <-- headline")
        print(f"               max|dVm| {r[7]:.3e} m/s, "
              f"max|d(ghost-marker offset)| {r[8]:.3e} m = {r[8] / G.hx:.2e} h"
              f"  (the spring coupling itself)")
        print(f"               ghost displacement of the reference at this "
              f"frame {r[9]:.4f} m ({r[9] / G.hx:.1f} h)")
        for m, ids in _mass_groups(R):
            da = np.linalg.norm(A["Xm"][keep[-1]][ids] - R["Xm"][keep[-1]][ids],
                                axis=1).max()
            # rigid-body metric: how far the group's centroid wanders apart
            dcen = max(np.linalg.norm(A["Xm"][k][ids].mean(axis=0)
                                      - R["Xm"][k][ids].mean(axis=0))
                       for k in keep)
            print(f"               M = {m:<9g} ({len(ids):3d} ghosts): "
                  f"max|dXm| {da:.3e} m = {da / G.hx:.2e} h, "
                  f"centroid drift <= {dcen:.3e} m = {dcen / G.hx:.2e} h")
    if has_c:
        print(f"concentration: max|dC| {r[-2]:.3e}, relative L2 {r[-1]:.3e}")


A = np.load(f_afsi)
R = np.load(f_ref)
G = Grid(int(A["nx"]), int(A["ny"]), float(A["dt"]))
keep = align(A, R)
rows = metrics(G, A, R, keep)
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
    Ax, Rx = np.load(afsi_path), np.load(ref_path)
    Gx = Grid(int(Ax["nx"]), int(Ax["ny"]), float(Ax["dt"]))
    kx = align(Ax, Rx)
    report(f"cross-check {tag}", Gx, Ax, Rx, kx, metrics(Gx, Ax, Rx, kx))

t = A["t"][keep]
has_m = "Xm" in A.files and A["Xm"].size
has_c = "C" in A.files and A["C"].size

with open(os.path.join(out, "compare_table.csv"), "w") as fh:
    fh.write(",".join(COLS[:5])
             + ("," + ",".join(COLS[5:10]) if has_m else "")
             + ("," + ",".join(COLS[10:]) if has_c else "") + "\n")
    for i, r in enumerate(rows):
        n = 5 + (5 if has_m else 0) + (2 if has_c else 0)
        fh.write(",".join(f"{v:.10g}" for v in r[:n]) + "\n")

# --- 1. the headline: mass-point (ghost) trajectories -----------------------
if has_m:
    groups = _mass_groups(R)
    fig, ax = plt.subplots(2, 3, figsize=(15.5, 8.0))
    Ma, Mr = A["Xm"], R["Xm"]
    for c, (m, ids) in enumerate(groups[:3]):
        # mean position of the ghosts of this mass group over the aligned frames
        Mag = Ma[np.ix_(keep, ids)]
        Mrg = Mr[np.ix_(keep, ids)]
        ax[0, c].plot(Mrg[:, :, 0].mean(axis=1), Mrg[:, :, 1].mean(axis=1),
                      "k-", lw=2.0, alpha=0.55, label="IB2d")
        ax[0, c].plot(Mag[:, :, 0].mean(axis=1), Mag[:, :, 1].mean(axis=1),
                      "r--", lw=1.4, label="AFSI")
        ax[0, c].plot(Mrg[0, :, 0].mean(), Mrg[0, :, 1].mean(),
                      "g*", ms=12, label="start")
        ax[0, c].set_title(f"ghost centroid, M = {m:g} ({len(ids)} pts)")
        ax[0, c].set_xlabel("x")
        ax[0, c].set_ylabel("y")
        ax[0, c].set_aspect("equal")
        ax[0, c].legend(fontsize=8)
        d = np.linalg.norm(Ma[keep[-1]][ids] - Mr[keep[-1]][ids], axis=1)
        ax[1, c].plot(t, [np.linalg.norm(Ma[k][ids] - Mr[k][ids], axis=1).max()
                          for k in keep], "b-o", ms=3, label="max")
        ax[1, c].plot(t, [np.linalg.norm(Ma[k][ids] - Mr[k][ids], axis=1).mean()
                          for k in keep], "g-s", ms=3, label="mean")
        ax[1, c].set_yscale("log")
        ax[1, c].set_title(f"|Δ ghost position|, M = {m:g} "
                           f"(last {d.max():.2e} m)")
        ax[1, c].set_xlabel("t")
        ax[1, c].legend(fontsize=8)
    fig.suptitle("massive (ghost) boundary points: trajectories (headline metric)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "compare_trajectories.png"), dpi=150)

# --- 2. histories -----------------------------------------------------------
npan = 3 + (1 if has_m else 0) + (1 if has_c else 0)
fig, axs = plt.subplots(1, npan, figsize=(4.4 * npan, 3.6))
axs = np.atleast_1d(axs)
i = 0
axs[i].plot(t, [r[1] for r in rows], "b-o", ms=3)
axs[i].set_title("marker max|dX| (m)")
axs[i].set_xlabel("t")
i += 1
axs[i].plot(t, [r[2] for r in rows], "r--", label="AFSI")
axs[i].plot(t, [r[3] for r in rows], "k-", lw=2, label="IB2d")
axs[i].set_title(r"fluid $\max|u|$")
axs[i].set_xlabel("t")
axs[i].legend(fontsize=8)
i += 1
axs[i].semilogy(t, [max(r[4], 1e-18) for r in rows], "b-o", ms=3)
axs[i].set_title(r"$\max|\Delta u|$")
axs[i].set_xlabel("t")
if has_m:
    i += 1
    axs[i].semilogy(t, [max(r[5], 1e-18) for r in rows], "b-o", ms=3,
                    label="max")
    axs[i].semilogy(t, [max(r[6], 1e-18) for r in rows], "g-s", ms=3,
                    label="rms")
    axs[i].set_title("ghost position error (m)")
    axs[i].set_xlabel("t")
    axs[i].legend(fontsize=8)
if has_c:
    i += 1
    axs[i].semilogy(t, [max(r[-1], 1e-18) for r in rows], "b-o", ms=3)
    axs[i].set_title("concentration relative $L_2$")
    axs[i].set_xlabel("t")
fig.tight_layout()
fig.savefig(os.path.join(out, "compare_history.png"), dpi=150)

# --- 3. fields at the last aligned frame ------------------------------------
k = keep[-1]
if has_c:
    Ca, Cr = A["C"][k], R["C"][k]
    dC = Ca - Cr
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    vmax = max(Ca.max(), Cr.max())
    for a_, C, title in ((ax[0], Cr, "IB2d"), (ax[1], Ca, "AFSI")):
        c = a_.pcolormesh(G.XNODES, G.YNODES, C, shading="auto", cmap="inferno",
                          vmin=0, vmax=vmax)
        a_.set_title(f"{title}: C, t = {t[-1]:.5f}")
        fig.colorbar(c, ax=a_, shrink=0.85)
    lim = max(np.abs(dC).max(), 1e-18)
    c = ax[2].pcolormesh(G.XNODES, G.YNODES, dC, shading="auto", cmap="RdBu_r",
                         vmin=-lim, vmax=lim)
    ax[2].set_title(f"$C^{{AFSI}} - C^{{IB2d}}$, max {lim:.2e}")
    fig.colorbar(c, ax=ax[2], shrink=0.85)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "compare_fields.png"), dpi=150)
else:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    vmax = max(np.abs(R["u"][k]).max(), np.abs(A["u"][k]).max())
    for a_, u, title in ((ax[0], R["u"][k], "IB2d"), (ax[1], A["u"][k], "AFSI")):
        c = a_.pcolormesh(G.XNODES, G.YNODES, np.linalg.norm(u, axis=-1),
                          shading="auto", cmap="viridis", vmin=0, vmax=vmax)
        a_.plot(R["X"][k][:, 0], R["X"][k][:, 1], "w.", ms=0.5)
        if has_m:
            a_.plot(R["Xm"][k][:, 0], R["Xm"][k][:, 1], "r.", ms=2)
        a_.set_title(f"{title}: |u|, t = {t[-1]:.5f}")
        fig.colorbar(c, ax=a_, shrink=0.85)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "compare_fields.png"), dpi=150)

print(f"figures written to {out}")
