"""Compare the AFSI poroelastic rubberband with the pyIB2d reference, and verify
the AFSI restart protocol against the AFSI continuous run.

    python compare.py [ib2d_reference.npz] [out_dir] [--restart-at K]
                      [--restart-tend T] [--skip-restart]

Inputs: ``plot/afsi_result.npz`` (main.py) and ``ib2d_reference.npz``
(run_reference.py).  The structure is a closed ring of 64 markers held by
zero-rest-length linear springs (``k = 1e6``) whose markers additionally creep
with ``dt * F_spring / (mu c)`` (poroelastic, ``c = 2.5e5``); the ring
collapses and the quantities of interest are the enclosed area and the
marker-wise difference to the reference.

Frame alignment: the pyIB2d driver dumps when its 0-based step counter ``cter``
is a positive multiple of ``print_dump``, i.e. at the *end* of step
``cter + 1``, and stamps the frame with ``cter * dt`` -- one step behind the
state it writes (an upstream labelling quirk).  ``main.py`` dumps on exactly
those steps, so frame ``k`` of the two codes is the same step and
``t_afsi[k] = t_ref[k] + dt``.

Restart check: ``main.py`` is run three times --

  1. continuously to ``T`` (or the existing ``plot/afsi_result.npz`` is used);
  2. to ``t1 = K*dt`` with ``RESTART_DUMP`` writing the full solver state;
  3. from that state to ``T`` with ``RESTART_FROM``.

The restarted trajectory (2+3) must reproduce run 1 to round-off; the deviation
is reported together with a *noise floor* measured from the same data: the
frames of run 2 with ``t <= t1`` are the same dynamics computed by an
independent process as run 1, so their difference is the process-level
reproducibility limit of the pipeline (BLAS reductions are not bitwise
reproducible).  ``--noise-floor`` additionally re-runs the full continuous
configuration as a fourth process and reports its deviation over all frames.
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
import ib2d_io  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("ref", nargs="?", default=os.path.join(here, "ib2d_reference.npz"))
ap.add_argument("out", nargs="?", default=os.path.join(here, "figures"))
ap.add_argument("--restart-at", type=int, default=None,
                help="completed steps at which the state is dumped (K)")
ap.add_argument("--restart-tend", type=float, default=None,
                help="final time of the restart check (default Tfinal)")
ap.add_argument("--skip-restart", action="store_true")
ap.add_argument("--noise-floor", action="store_true",
                help="also re-run the full continuous config and report its "
                     "deviation (extra full run)")
args = ap.parse_args()

f_ref, out = args.ref, args.out
os.makedirs(out, exist_ok=True)

P = ib2d_io.read_input2d(os.path.join(here, "ib2d_input", "input2d"))
dt = float(P["dt"])
h = float(P["Lx"]) / int(P["Nx"])           # grid spacing
T_full = float(P["Tfinal"])
print_dump = int(P["print_dump"])

f_afsi = os.path.join(here, "plot", "afsi_result.npz")

R = np.load(f_ref)
A = np.load(f_afsi)
n = min(len(A["t"]), len(R["t"]))
assert A["X"].shape[1] == R["X"].shape[1] == 64, "marker counts differ"

t = A["t"][:n]                  # true times of the AFSI (and reference) frames
Xa, Xr = A["X"][:n], R["X"][:n]

dX = np.linalg.norm(Xa - Xr, axis=2).max(axis=1)
dcx = np.abs(Xa[:, :, 0].mean(axis=1) - Xr[:, :, 0].mean(axis=1))
dcy = np.abs(Xa[:, :, 1].mean(axis=1) - Xr[:, :, 1].mean(axis=1))


def ring_area(Xm):
    x, y = Xm[..., 0], Xm[..., 1]
    return 0.5 * np.sum(x * np.roll(y, -1, axis=-1) - y * np.roll(x, -1, axis=-1),
                        axis=-1)


def ring_perimeter(Xm):
    x, y = Xm[..., 0], Xm[..., 1]
    return np.sum(np.hypot(np.roll(x, -1, axis=-1) - x,
                           np.roll(y, -1, axis=-1) - y), axis=-1)


aa, ar = ring_area(Xa), ring_area(Xr)
pa, pr = ring_perimeter(Xa), ring_perimeter(Xr)
cx_a, cy_a = Xa[:, :, 0].mean(axis=1), Xa[:, :, 1].mean(axis=1)

print(f"frames: AFSI {len(A['t'])}, IB2d {len(R['t'])}, compared {n} "
      f"(t = {t[0]:.5f} .. {t[-1]:.5f}, dt = {dt:g})")
print(f"{'t':>7} {'dXmax':>9} {'dXmax/h':>8} {'dCentroid':>10} "
      f"{'area A/I':>17} {'perim A/I':>17}")
rows = []
for k in range(n):
    rows.append((t[k], dX[k], dX[k] / h, max(dcx[k], dcy[k]),
                 aa[k], ar[k], pa[k], pr[k]))
    if k % max(1, n // 12) == 0 or k == n - 1:
        print(f"{t[k]:7.4f} {dX[k]:9.3e} {dX[k] / h:8.3e} "
              f"{max(dcx[k], dcy[k]):10.3e} "
              f"{aa[k]:8.6f}/{ar[k]:8.6f} {pa[k]:8.6f}/{pr[k]:8.6f}")
with open(os.path.join(out, "compare_table.csv"), "w") as fh:
    fh.write("t,dXmax,dXmax_over_h,dcentroid,area_afsi,area_ib2d,"
             "perimeter_afsi,perimeter_ib2d\n")
    for row in rows:
        fh.write(",".join(f"{v:.10g}" for v in row) + "\n")

print(f"\nmarker deviation : max {dX.max():.3e} m = {dX.max() / h:.2e} h "
      f"(h = {h:g}); typical {np.median(dX[1:]):.2e} m")
print(f"centroid         : max |dc| = {max(dcx.max(), dcy.max()):.3e} m")
print(f"enclosed area    : AFSI {aa[0]:.6f} -> {aa[-1]:.6f} (drift "
      f"{abs(aa[-1] - aa[0]):.2e}), IB2d {ar[0]:.6f} -> {ar[-1]:.6f} (drift "
      f"{abs(ar[-1] - ar[0]):.2e}); max |dA| = {np.abs(aa - ar).max():.2e}")
print(f"perimeter        : AFSI {pa[0]:.6f} -> {pa[-1]:.6f}, IB2d "
      f"{pr[0]:.6f} -> {pr[-1]:.6f}; max |dP| = {np.abs(pa - pr).max():.2e}")

# ---------------------------------------------------------------------------
# Restart protocol: continuous run vs (segment + restart)
# ---------------------------------------------------------------------------
restart_dev = None
if not args.skip_restart:
    T_rt = args.restart_tend if args.restart_tend is not None else T_full
    n_rt = int(round(T_rt / dt))
    K = (args.restart_at if args.restart_at is not None
         else print_dump * max(1, (n_rt // (2 * print_dump))) + 1)
    t1 = K * dt
    assert 0 < K < n_rt, "restart point must be inside (0, T)"
    seg1 = os.path.join(here, "plot_restart_seg1")
    rest = os.path.join(here, "plot_restart")
    state = os.path.join(here, "ib2d_run", "restart_state.npz")
    os.makedirs(os.path.dirname(state), exist_ok=True)

    def run_main(env):
        e = dict(os.environ)
        e.update(env)
        subprocess.run([sys.executable, os.path.join(here, "main.py")],
                       cwd=here, env=e, check=True)

    print(f"\nrestart check    : K={K} steps (t1={t1:g}), T={T_rt:g}, dt={dt:g}")
    run_main({"TFINAL": repr(t1), "RESTART_AT": str(K),
              "RESTART_DUMP": state, "OUTPUT_PATH": seg1})
    assert os.path.exists(state), "segment run wrote no restart state"
    run_main({"TFINAL": repr(T_rt), "RESTART_FROM": state,
              "OUTPUT_PATH": rest})

    S = np.load(os.path.join(seg1, "afsi_result.npz"))
    C = np.load(os.path.join(rest, "afsi_result.npz"))
    assert abs(float(C["t"][0]) - t1) < 1e-12, "restart run does not start at t1"

    # continuous reference for the check (same T_rt)
    if abs(T_rt - T_full) < 1e-12 and os.path.exists(f_afsi):
        cont = dict(np.load(f_afsi))
    else:
        print("                 (rerunning the continuous reference for "
              f"T={T_rt:g})")
        cont_dir = os.path.join(here, "plot_restart_cont")
        run_main({"TFINAL": repr(T_rt), "OUTPUT_PATH": cont_dir})
        cont = dict(np.load(os.path.join(cont_dir, "afsi_result.npz")))

    tc = np.asarray(cont["t"])
    sel_c = tc <= T_rt + 1e-12
    tc, Xc, uc, pc = (tc[sel_c], cont["X"][sel_c], cont["u"][sel_c],
                      cont["p"][sel_c])

    def compare_frames(tq, Xq, uq, pq, label):
        idx = np.array([int(np.argmin(np.abs(tc - x))) for x in tq])
        assert np.allclose(tc[idx], tq, atol=1e-12), "frame times differ"
        dX = np.linalg.norm(Xq - Xc[idx], axis=2).max(axis=1)
        du = np.abs(uq - uc[idx]).max(axis=(1, 2, 3))
        dp = np.abs(pq - pc[idx]).max(axis=(1, 2))
        print(f"{label:19s}: {len(tq)} frames "
              f"t={tq[0]:.4f}..{tq[-1]:.4f}")
        print(f"{'':19s}  max |dX|={dX.max():.3e} m ({dX.max() / h:.2e} h), "
              f"max |du|={du.max():.3e}, max |dp|={dp.max():.3e}")
        return dX, du, dp

    def frame_dev(Xq, uq, pq, idx):
        return (np.linalg.norm(Xq - Xc[idx], axis=2).max(axis=1),
                np.abs(uq - uc[idx]).max(axis=(1, 2, 3)),
                np.abs(pq - pc[idx]).max(axis=(1, 2)))

    # noise floor: the segment run is the same dynamics (0..t1) computed by an
    # independent process, so seg-vs-cont there is the reproducibility limit.
    ms = np.asarray(S["t"]) <= t1 + 1e-12
    idx_s = np.array([int(np.argmin(np.abs(tc - x))) for x in S["t"][ms]])
    dX_s, du_s, dp_s = frame_dev(S["X"][ms], S["u"][ms], S["p"][ms], idx_s)
    print(f"{'noise floor (seg)':19s}: {int(ms.sum())} frames "
          f"t={S['t'][ms][0]:.4f}..{S['t'][ms][-1]:.4f}")
    print(f"{'':19s}  max |dX|={dX_s.max():.3e} m "
          f"({dX_s.max() / h:.2e} h), max |du|={du_s.max():.3e}, "
          f"max |dp|={dp_s.max():.3e}")

    mr = np.asarray(C["t"]) >= t1 - 1e-12
    dX_rt, du_rt, dp_rt = compare_frames(C["t"][mr], C["X"][mr], C["u"][mr],
                                         C["p"][mr], "restart vs cont")
    tcmp = np.asarray(C["t"])[mr]
    restart_dev = (dX_rt.max(), du_rt.max(), dp_rt.max())
    print(f"{'':19s}  continuous-vs-restart at T: |dX|={dX_rt[-1]:.3e}, "
          f"|du|={du_rt[-1]:.3e} (noise floor over 0..t1 "
          f"{dX_s.max():.3e})")

    if args.noise_floor:
        cont2 = os.path.join(here, "plot_restart_cont2")
        run_main({"TFINAL": repr(T_rt), "OUTPUT_PATH": cont2})
        D = np.load(os.path.join(cont2, "afsi_result.npz"))
        idx_d = np.array([int(np.argmin(np.abs(tc - x))) for x in D["t"]])
        dX_d, du_d, dp_d = frame_dev(D["X"], D["u"], D["p"], idx_d)
        print(f"{'noise floor (rerun)':19s}: {len(D['t'])} frames over 0..T")
        print(f"{'':19s}  max |dX|={dX_d.max():.3e} m, "
              f"max |du|={du_d.max():.3e}, max |dp|={dp_d.max():.3e}")

    with open(os.path.join(out, "restart_table.csv"), "w") as fh:
        fh.write("t,dXmax,du_max,dp_max\n")
        for k in range(len(tcmp)):
            fh.write(f"{tcmp[k]:.10g},{dX_rt[k]:.10g},{du_rt[k]:.10g},"
                     f"{dp_rt[k]:.10g}\n")

# --- 1. band + anchors at several times -------------------------------------
nsel = 6
sel = np.linspace(0.0, t[-1], nsel)
fig, axes = plt.subplots(1, nsel, figsize=(2.6 * nsel, 5.4))
lim = (Xr[..., 0].min() - 0.05, Xr[..., 0].max() + 0.05)
vlim = (Xr[..., 1].min() - 0.05, Xr[..., 1].max() + 0.05)
for ax, tsel in zip(np.atleast_1d(axes), sel):
    k = int(np.argmin(abs(t - tsel)))
    ax.plot(np.append(Xr[k, :, 0], Xr[k, 0, 0]),
            np.append(Xr[k, :, 1], Xr[k, 0, 1]), "k-", lw=2.0, alpha=0.55,
            label="IB2d")
    ax.plot(np.append(Xa[k, :, 0], Xa[k, 0, 0]),
            np.append(Xa[k, :, 1], Xa[k, 0, 1]), "r--", lw=1.2, label="AFSI")
    ax.set_title(f"t = {t[k]:.4f}")
    ax.set_aspect("equal")
    ax.set_xlim(*lim)
    ax.set_ylim(*vlim)
np.atleast_1d(axes)[0].legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(out, "compare_shapes.png"), dpi=150)

# --- 2. histories -----------------------------------------------------------
fig, ax = plt.subplots(1, 4, figsize=(17, 3.4))
ax[0].semilogy(t[1:], dX[1:], "k-o", ms=3, label="markers")
ax[0].semilogy(t[1:], np.maximum(dcx[1:], 1e-16), "b-s", ms=3,
               label="centroid (x)")
ax[0].axhline(h, color="gray", ls=":", label="grid spacing h")
ax[0].axhline(np.median(dX[1:]), color="g", ls=":", label="median")
ax[0].set_title(r"deviation vs IB2d")
ax[0].set_xlabel("t")
ax[0].legend(fontsize=8)
ax[1].plot(t, aa, "k-", lw=2, label="IB2d")
ax[1].plot(t, ar, "r--", label="AFSI")
ax[1].set_title("enclosed area (shoelace)")
ax[1].set_xlabel("t")
ax[1].legend(fontsize=8)
ax[2].plot(t, pa, "k-", lw=2, label="IB2d")
ax[2].plot(t, pr, "r--", label="AFSI")
ax[2].set_title("perimeter (relaxation towards a circle)")
ax[2].set_xlabel("t")
ax[2].legend(fontsize=8)
ax[3].plot(t, Xr[:, :, 0].min(axis=1), "k-", lw=2, label="IB2d $x_{min}$")
ax[3].plot(t, Xa[:, :, 0].min(axis=1), "r--", label="AFSI $x_{min}$")
ax[3].plot(t, Xr[:, :, 0].max(axis=1), "k:", lw=2, label="IB2d $x_{max}$")
ax[3].plot(t, Xa[:, :, 0].max(axis=1), "r-.", label="AFSI $x_{max}$")
ax[3].set_title("ring extents")
ax[3].set_xlabel("t")
ax[3].legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(out, "compare_history.png"), dpi=150)

# --- 3. fields at the last common frame -------------------------------------
if "u" in R and "u" in A:
    k = n - 1
    ua, ur = A["u"][k], R["u"][k]
    ny, nx = ur.shape[:2]
    xg = np.arange(nx) * float(R["dx"])
    yg = np.arange(ny) * float(R["dy"])

    def vort(u):
        duy = np.gradient(u[..., 0], yg, axis=0)
        dvx = np.gradient(u[..., 1], xg, axis=1)
        return dvx - duy

    wa_, wr_ = vort(ua), vort(ur)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    vmax = max(abs(wr_).max(), abs(wa_).max())
    for a_, w, X, title in ((ax[0], wr_, Xr, "IB2d"),
                            (ax[1], wa_, Xa, "AFSI")):
        c = a_.contourf(xg, yg, w, 40, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        a_.plot(np.append(X[k, :, 0], X[k, 0, 0]),
                np.append(X[k, :, 1], X[k, 0, 1]), "k-", lw=1.0)
        a_.set_title(f"{title}: vorticity, t = {t[k]:.4f}")
        a_.set_aspect("equal")
        fig.colorbar(c, ax=a_, shrink=0.8)
    c = ax[2].contourf(xg, yg, np.linalg.norm(ua - ur, axis=-1), 30,
                       cmap="viridis")
    ax[2].set_title(r"$|u^{AFSI}-u^{IB2d}|$, "
                    f"max {np.linalg.norm(ua - ur, axis=-1).max():.2e}")
    ax[2].set_aspect("equal")
    fig.colorbar(c, ax=ax[2], shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "compare_fields.png"), dpi=150)
    print(f"velocity         : max|u| AFSI "
          f"{np.linalg.norm(ua, axis=-1).max():.3f}, IB2d "
          f"{np.linalg.norm(ur, axis=-1).max():.3f}; last-frame "
          f"max|du| {np.linalg.norm(ua - ur, axis=-1).max():.2e}")

# --- 4. restart history ------------------------------------------------------
if restart_dev is not None and os.path.exists(os.path.join(out, "restart_table.csv")):
    tab = np.loadtxt(os.path.join(out, "restart_table.csv"), delimiter=",",
                     skiprows=1)
    fig, ax = plt.subplots(1, 1, figsize=(6, 3.6))
    ax.semilogy(tab[:, 0], np.maximum(tab[:, 1], 1e-17), "k-o", ms=3,
                label=r"$\max_k|X^{restart}-X^{cont}|$")
    ax.set_xlabel("t")
    ax.set_title("restart vs continuous run")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "compare_restart.png"), dpi=150)

print(f"figures written to {out}")
