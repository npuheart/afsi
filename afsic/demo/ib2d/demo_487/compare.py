"""Compare an AFSI port with the pyIB2d reference.

    python compare.py [ib2d_reference.npz] [out_dir]

Inputs: ``plot/afsi_result.npz`` (main.py) and ``ib2d_reference.npz``
(run_reference.py).  Everything that can be derived from the example's own
``input2d``/``ib2d_input`` is, so the same script serves every demo generated
from ``_template``:

* the markers, the velocity/pressure/force fields and -- when present -- the
  tracers and the concentration scalar are compared frame by frame;
* the four-node delta-function band of the Lagrangian structure (and, when the
  example has an arbitrary-external-force module, its forcing window) is
  reconstructed from the example data, and the velocity error is reported both
  inside and outside the band;
* the structure scalars (mean/max spring length, the target-anchor drift, the
  y-extent of the structure) come from ``ib2d_input``;
* when the example ships ``please_Compute_External_Forcing.py``, the module is
  evaluated on the AFSI velocity and its force is checked against the AFSI
  force field (they must agree on the window nodes away from the IB band).

Frame alignment: the pyIB2d driver dumps when its 0-based step counter
``cter`` is a positive multiple of ``print_dump`` -- i.e. at the end of step
``cter + 1`` -- and stamps that frame with ``cter * dt``, one step behind the
state it writes (upstream labelling quirk).  ``main.py`` dumps on exactly those
steps and stamps the true time, so AFSI frame ``k`` and reference frame ``k``
are the same step whenever ``round(t_afsi/dt) == round(t_ref/dt) + 1``; that is
checked here and only matching frames are used.

``ib2d_reference<TAG>.npz`` is paired with ``plot<TAG>/afsi_result.npz`` and
reported as a cross-check (used for dt/2 resolution studies).

Writes ``figures/compare_history.png``, ``figures/compare_fields.png``,
``figures/compare_profiles.png`` (and ``compare_concentration.png`` when the
example has a scalar) plus ``figures/compare_table.csv``.
"""
import glob
import importlib
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

sys.path.insert(0, here)
import ib2d_io  # noqa: E402

SRC = os.path.join(here, "ib2d_input")
P = ib2d_io.read_input2d(os.path.join(SRC, "input2d"))
EX = ib2d_io.load_example(SRC)
Lx, Ly = float(P["Lx"]), float(P["Ly"])
X0 = EX["X"]


def user_module(name):
    path = os.path.join(SRC, name + ".py")
    if not os.path.exists(path):
        return None
    sys.path.insert(0, SRC)
    try:
        return importlib.import_module(name)
    finally:
        sys.path.remove(SRC)


EXT = user_module("please_Compute_External_Forcing")


class Grid:
    """Node grid of one (AFSI, reference) pair plus the derived masks."""

    def __init__(self, nx, ny, dt, anchors, nb):
        self.nx, self.ny = nx, ny
        self.hx, self.hy = Lx / nx, Ly / ny
        self.dt = dt
        self.XNODES = np.arange(nx) * self.hx
        self.YNODES = np.arange(ny) * self.hy
        self.GRID_INFO = [nx, ny, Lx, Ly, self.hx, self.hy, int(P["supp"]), nb,
                          Lx / (2 * nx)]
        # four-point delta support of every anchor (marker or target point);
        # anchors are folded into the periodic box first (IB2d does the same)
        self.band = np.zeros((ny, nx), bool)
        for x0, y0 in anchors:
            i0 = int(round((x0 % Lx) / self.hx))
            j0 = int(round((y0 % Ly) / self.hy))
            self.band[max(0, j0 - 2):j0 + 3, max(0, i0 - 2):i0 + 3] = True
        self.away = ~self.band
        # the example's own forcing window, evaluated on this grid
        self.win = None
        if EXT is not None and hasattr(EXT, "_indices_to_apply_force"):
            self.win = np.zeros((ny, nx), bool)
            i, j = EXT._indices_to_apply_force(self.XNODES, self.YNODES).T
            self.win[j, i] = True
            self.core = self.win & self.away
            self.col = self.win.any(axis=1) & ~self.band[:, nx // 2]

    def module_force(self, t, ux, uy):
        """(Fx, Fy) of the ported arbitrary-external-force module."""
        return EXT.please_Compute_External_Forcing(
            self.dt, t, self.XNODES, self.YNODES, self.GRID_INFO, ux.copy(),
            uy.copy(), 1, None)[:2]

    def target_velocity(self, t):
        """u^tar of the module, read back through Fx = -k (u - u^tar) at u=0."""
        fx = self.module_force(t, np.zeros((self.ny, self.nx)),
                               np.zeros((self.ny, self.nx)))[0]
        return -fx / getattr(EXT, "K_STIFF", 1.0)


def spring_lengths(X):
    """Mean/max length of the example's springs at configuration ``X``."""
    if "springs" not in EX:
        return None, None
    conn = EX["springs"]["conn"]
    L = np.linalg.norm(X[conn[:, 1]] - X[conn[:, 0]], axis=1)
    return L.mean(), L.max()


def metrics(G, A, R):
    dt = G.dt
    steps_a = np.rint(A["t"] / dt).astype(int)
    steps_r = np.rint(R["t"] / dt).astype(int)
    # the driver's pre-loop dump is the initial state (step 0); every later
    # frame k is the state at step steps_r[k] + 1
    keep = [k for k in range(min(len(steps_a), len(steps_r)))
            if steps_a[k] == steps_r[k] + (0 if k == 0 else 1)]
    assert keep, f"no aligned frames: AFSI {steps_a} vs reference {steps_r}"

    hasC = "C" in A.files and "C" in R.files
    hasT = "tracers" in A.files and "tracers" in R.files
    rows = []
    for k in keep:
        ua, ur = A["u"][k], R["u"][k]
        du = np.linalg.norm(ua - ur, axis=-1)
        nrm = np.linalg.norm(ur[G.away])
        d = dict(
            t=float(A["t"][k]),
            dXmax=float(np.linalg.norm(A["X"][k] - R["X"][k], axis=1).max()),
            umax_a=float(np.abs(ua).max()), umax_r=float(np.abs(ur).max()),
            du_away=float(du[G.away].max()), du_band=float(du[G.band].max()),
            relL2_away=float(np.linalg.norm(du[G.away]) / nrm) if nrm else 0.0,
            df_away=float(np.abs(A["f"][k] - R["f"][k])[G.away].max()),
            dXa=float("nan"), Lmean_a=float("nan"), Lmean_r=float("nan"),
            umax_win_a=float("nan"), umax_win_r=float("nan"),
            dtracer=float("nan"), dCmax=float("nan"), dC_relL2=float("nan"),
            Cmax_a=float("nan"), Cmax_r=float("nan"), Csum_a=float("nan"),
            Csum_r=float("nan"), dC_away=float("nan"), dC_band=float("nan"),
            f_res=float("nan"),
        )
        if "Xa" in A.files and "Xa" in R.files and EX.get("targets") is not None:
            d["dXa"] = float(np.abs(A["Xa"][k] - R["Xa"][k]).max())
        la, _ = spring_lengths(A["X"][k])
        lr, _ = spring_lengths(R["X"][k])
        if la is not None:
            d["Lmean_a"], d["Lmean_r"] = float(la), float(lr)
        if G.win is not None:
            d["umax_win_a"] = float(np.abs(ua)[G.win].max())
            d["umax_win_r"] = float(np.abs(ur)[G.win].max())
            f_mod = G.module_force(d["t"], ua[..., 0], ua[..., 1])[0]
            res = np.abs(A["f"][k][..., 0] - f_mod)[G.core]
            d["f_res"] = float(res.max()) if res.size else 0.0
        if hasT:
            d["dtracer"] = float(np.linalg.norm(A["tracers"][k]
                                                - R["tracers"][k], axis=1).max())
        if hasC:
            Ca, Cr = A["C"][k], R["C"][k]
            dC = np.abs(Ca - Cr)
            d["dCmax"] = float(dC.max())
            nrmC = np.linalg.norm(Cr)
            d["dC_relL2"] = float(np.linalg.norm(Ca - Cr) / nrmC) if nrmC else 0.0
            d["Cmax_a"], d["Cmax_r"] = float(Ca.max()), float(Cr.max())
            d["Csum_a"] = float(Ca.sum())
            d["Csum_r"] = float(Cr.sum())
            d["dC_away"] = float(dC[G.away].max())
            d["dC_band"] = float(dC[G.band].max())
        rows.append(d)
    return keep, rows


COLS = ["t", "dXmax", "umax_a", "umax_r", "du_away", "du_band", "relL2_away",
        "df_away", "dXa", "Lmean_a", "Lmean_r", "umax_win_a", "umax_win_r",
        "dtracer", "dCmax", "dC_relL2", "Cmax_a", "Cmax_r", "Csum_a", "Csum_r",
        "dC_away", "dC_band", "f_res"]


def _rel(a, b):
    """Relative difference, 0 when the reference value vanishes."""
    return abs(a - b) / b if b else 0.0


def report(tag, G, A, R, keep, rows):
    r = rows[-1]
    print(f"\n=== {tag} (dt={G.dt:g}, h={G.hx:g}, {len(keep)} aligned frames, "
          f"t={rows[0]['t']:.5f}..{r['t']:.5f})")
    hdr = f"{'t':>9} {'dXmax/h':>9} {'max|u| a/r':>21} {'du away':>10} " \
          f"{'du band':>10} {'relL2':>10} {'df away':>10}"
    if "tracers" in A.files:
        hdr += f" {'dtracer':>10}"
    if "C" in A.files:
        hdr += f" {'max|dC|':>10} {'relL2(C)':>10}"
    print(hdr)
    for x in rows:
        line = (f"{x['t']:9.5f} {x['dXmax'] / G.hx:9.2e} "
                f"{x['umax_a']:10.3e}/{x['umax_r']:<10.3e} "
                f"{x['du_away']:10.3e} {x['du_band']:10.3e} "
                f"{x['relL2_away']:10.3e} {x['df_away']:10.3e}")
        if "tracers" in A.files:
            line += f" {x['dtracer']:10.3e}"
        if "C" in A.files:
            line += f" {x['dCmax']:10.3e} {x['dC_relL2']:10.3e}"
        print(line)
    print(f"markers          : max|dX| {r['dXmax']:.3e} m = "
          f"{r['dXmax'] / G.hx:.2e} h "
          f"({A['X'].shape[1]} Lagrangian points)")
    if EX.get("targets") is not None and not np.isnan(r["dXa"]):
        print(f"target anchors   : max|Xa_AFSI - Xa_IB2d| {r['dXa']:.3e} m = "
              f"{r['dXa'] / G.hx:.2e} h (the example's update module moves "
              f"them in both codes)")
    print(f"velocity         : max|u| AFSI {r['umax_a']:.4e} / IB2d "
          f"{r['umax_r']:.4e} ({_rel(r['umax_a'], r['umax_r']):.2e} "
          f"relative); max|du| {r['du_away']:.3e} = "
          f"{_rel(r['du_away'], r['umax_r']):.2e} max|u| away from the IB band "
          f"({int(G.away.sum())}/{G.away.size} nodes), {r['du_band']:.3e} "
          f"({_rel(r['du_band'], r['umax_r']):.2f} max|u|) inside it; "
          f"relative L2 {r['relL2_away']:.3e}")
    print(f"force field      : max|df| {r['df_away']:.3e} away from the band, "
          f"|f|max {np.abs(R['f'][keep[-1]]).max():.3e}")
    if G.win is not None:
        print(f"forcing window   : {int(G.win.sum())} nodes, max|u| AFSI "
              f"{r['umax_win_a']:.4f} / IB2d {r['umax_win_r']:.4f} "
              f"({_rel(r['umax_win_a'], r['umax_win_r']):.2e} "
              f"relative); forcing check |f_x - F_x^module(u^AFSI)| <= "
              f"{r['f_res']:.3e} on the core nodes")
    if not np.isnan(r["Lmean_a"]):
        print(f"springs          : mean length AFSI {r['Lmean_a']:.6f} / IB2d "
              f"{r['Lmean_r']:.6f} "
              f"({_rel(r['Lmean_a'], r['Lmean_r']):.2e} relative)")
    if "tracers" in A.files:
        tr = [x["dtracer"] for x in rows]
        print(f"tracers          : max|dX_t| {max(tr):.3e} m = "
              f"{max(tr) / G.hx:.2e} h ({A['tracers'].shape[1]} passive points)")
    if "C" in A.files:
        print(f"concentration    : max|dC| {r['dCmax']:.3e}, relative L2 "
              f"{r['dC_relL2']:.3e}; peak AFSI {r['Cmax_a']:.6f} / IB2d "
              f"{r['Cmax_r']:.6f} ({_rel(r['Cmax_a'], r['Cmax_r']):.2e}"
              f" relative); total AFSI {r['Csum_a']:.8f} / IB2d "
              f"{r['Csum_r']:.8f}")
        print(f"                   max|dC| {r['dC_away']:.3e} away from the "
              f"band, {r['dC_band']:.3e} inside it")


def load_pair(ref_path, afsi_path):
    A = np.load(afsi_path)
    R = np.load(ref_path)
    dt = float(A["dt"]) if "dt" in A.files else float(
        R["dt"] if "dt" in R.files else P["dt"])
    ny, nx = A["u"].shape[1:3]
    # the IB band is the four-point delta support of *every* marker (target
    # anchors included, as an update module may move them): use the initial
    # configuration of both
    anchors = A["X"][0]
    if "Xa" in A.files and len(A["Xa"][0]):
        anchors = np.vstack([A["Xa"][0], anchors])
    G = Grid(nx, ny, dt, anchors, A["X"].shape[1])
    return G, A, R


f_afsi = os.path.join(here, "plot", "afsi_result.npz")
G, A, R = load_pair(f_ref, f_afsi)
keep, rows = metrics(G, A, R)
report("shipped setup", G, A, R, keep, rows)

# cross-checks: ``ib2d_reference<TAG>.npz`` pairs with ``plot<TAG>/``
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

with open(os.path.join(out, "compare_table.csv"), "w") as fh:
    fh.write(",".join(COLS) + "\n")
    for x in rows:
        fh.write(",".join(f"{x[c]:.10g}" for c in COLS) + "\n")

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
t = np.array([x["t"] for x in rows])
dX = np.array([x["dXmax"] for x in rows])
dXa = np.array([x["dXa"] for x in rows])

fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.6))
ax[0].semilogy(t, np.maximum(dX, 1e-16), "-o", ms=3, label="max|dX|")
if not np.all(np.isnan(dXa)):
    ax[0].semilogy(t, np.maximum(dXa, 1e-16), "--", label="max|dXa|")
ax[0].set(xlabel="t (s)", ylabel="m", title=f"markers (h={G.hx:.3g} m)")
ax[0].grid(True, which="both", alpha=.3)
ax[0].legend()
ax[1].plot(t, [x["umax_a"] for x in rows], "-o", ms=3, label="AFSI")
ax[1].plot(t, [x["umax_r"] for x in rows], "--", label="IB2d")
ax[1].set(xlabel="t (s)", ylabel="max|u| (m/s)", title="velocity amplitude")
ax[1].grid(alpha=.3)
ax[1].legend()
ax[2].semilogy(t, [x["du_away"] for x in rows], "-o", ms=3, label="away")
ax[2].semilogy(t, [x["du_band"] for x in rows], "--", label="IB band")
if "C" in A.files:
    ax[2].semilogy(t, [x["dCmax"] for x in rows], "-.", label="max|dC|")
ax[2].set(xlabel="t (s)", ylabel="error", title="velocity / scalar error")
ax[2].grid(True, which="both", alpha=.3)
ax[2].legend()
fig.tight_layout()
fig.savefig(os.path.join(out, "compare_history.png"), dpi=140)
plt.close(fig)

k = keep[-1]
fig, ax = plt.subplots(2, 3, figsize=(13.5, 7.4))
im = ax[0, 0].imshow(np.linalg.norm(A["u"][k], axis=-1), origin="lower",
                     extent=[0, Lx, 0, Ly])
plt.colorbar(im, ax=ax[0, 0])
ax[0, 0].set_title(f"|u| AFSI, t={A['t'][k]:.4f}")
im = ax[0, 1].imshow(np.linalg.norm(R["u"][k], axis=-1), origin="lower",
                     extent=[0, Lx, 0, Ly])
plt.colorbar(im, ax=ax[0, 1])
ax[0, 1].set_title(f"|u| IB2d, t={A['t'][k]:.4f}")
im = ax[0, 2].imshow(np.linalg.norm(A["u"][k] - R["u"][k], axis=-1),
                     origin="lower", extent=[0, Lx, 0, Ly])
ax[0, 2].set_title("|u| AFSI - IB2d")
plt.colorbar(im, ax=ax[0, 2])
im = ax[1, 0].imshow(A["p"][k], origin="lower", extent=[0, Lx, 0, Ly])
plt.colorbar(im, ax=ax[1, 0])
ax[1, 0].set_title("p AFSI")
im = ax[1, 1].imshow(R["p"][k], origin="lower", extent=[0, Lx, 0, Ly])
plt.colorbar(im, ax=ax[1, 1])
ax[1, 1].set_title("p IB2d")
im = ax[1, 2].imshow(A["p"][k] - R["p"][k], origin="lower",
                     extent=[0, Lx, 0, Ly])
plt.colorbar(im, ax=ax[1, 2])
ax[1, 2].set_title("p AFSI - IB2d")
for a in ax.ravel():
    a.set_aspect("equal")
fig.tight_layout()
fig.savefig(os.path.join(out, "compare_fields.png"), dpi=140)
plt.close(fig)

# profiles along the centre column / row + structure overlay
fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))
i = G.nx // 2
ax[0].plot(A["u"][k, :, i, 0], G.YNODES, "-o", ms=3, label="AFSI")
ax[0].plot(R["u"][k, :, i, 0], G.YNODES, "--", label="IB2d")
ax[0].set(xlabel="u_x (m/s)", ylabel="y (m)", title=f"u_x at x={i * G.hx:.3g}")
ax[0].grid(alpha=.3)
ax[0].legend()
j = G.ny // 2
ax[1].plot(G.XNODES, A["u"][k, j, :, 1], "-o", ms=3, label="AFSI")
ax[1].plot(G.XNODES, R["u"][k, j, :, 1], "--", label="IB2d")
ax[1].set(xlabel="x (m)", ylabel="u_y (m/s)", title=f"u_y at y={j * G.hy:.3g}")
ax[1].grid(alpha=.3)
ax[1].legend()
ax[2].plot(X0[:, 0], X0[:, 1], "k-", lw=.6)
ax[2].plot(A["X"][k, :, 0], A["X"][k, :, 1], "C0-", lw=.9, label="AFSI")
ax[2].plot(R["X"][k, :, 0], R["X"][k, :, 1], "C1--", lw=.9, label="IB2d")
ax[2].set(xlabel="x (m)", ylabel="y (m)", title="structure")
ax[2].set_aspect("equal")
ax[2].grid(alpha=.3)
ax[2].legend()
fig.tight_layout()
fig.savefig(os.path.join(out, "compare_profiles.png"), dpi=140)
plt.close(fig)

if "C" in A.files:
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))
    vmin = min(A["C"][k].min(), R["C"][k].min())
    vmax = max(A["C"][k].max(), R["C"][k].max())
    for c, (F, nm) in enumerate(((A["C"][k], "C AFSI"),
                                 (R["C"][k], "C IB2d"))):
        im = ax[c].imshow(F, origin="lower", extent=[0, Lx, 0, Ly],
                          vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax[c])
        ax[c].set_title(f"{nm}, t={A['t'][k]:.4f}")
    im = ax[2].imshow(A["C"][k] - R["C"][k], origin="lower",
                      extent=[0, Lx, 0, Ly])
    plt.colorbar(im, ax=ax[2])
    ax[2].set_title("C AFSI - IB2d")
    for a in ax:
        a.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "compare_concentration.png"), dpi=140)
    plt.close(fig)

print(f"\nwrote {out}/compare_table.csv and the figures")
