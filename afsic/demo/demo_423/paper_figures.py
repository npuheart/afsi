#!/usr/bin/env python
"""Paper figures for demo_423 (immersed anisotropic annular solid).

Produces, into ``figures/``:

  fig1_convergence.png   log-log error vs h for IPCS and Chorin, with
                         reference slopes h^1, h^1.5, h^2
  fig2_pressure_profile.png
                         p(r) numerical vs analytical, and the pointwise
                         error, along the outward radius at y = 0.5
  fig3_dt_study.png      e_p_L2 vs dt at fixed N=64 (Chorin vs IPCS)
  fig4_error_map.png     2-D |p_h - p_exact| field (needs pressure_error.xdmf)

Inputs
------
  --ipcs-json    JSON written by ``convergence.py --solver ipcs``
  --chorin-json  JSON written by ``convergence.py --solver chorin``
  --profiles     glob of ``profile_N*.csv`` / ``ipcs*_profile_N*.csv``
  --error-field  XDMF holding the final ``pressure_error`` (optional)

Run from this directory inside the afsi-dolfinx environment:

    python paper_figures.py \
        --ipcs-json   ../../../../.conda/scratch/rerun_ipcs.json \
        --chorin-json ../../../../.conda/scratch/rerun_chorin.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Annulus
from matplotlib.ticker import NullFormatter
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R, W = 0.25, 0.0625          # inner radius, annular width
MU_S, LX = 1.0, 1.0


def exact_pressure(r):
    """Analytical pressure, zero-mean gauge handled by the caller."""
    const = -np.pi * MU_S / (2.0 * LX * LX) * ((R + W) ** 2 - R ** 2)
    tiny = 1.0e-14
    inside = MU_S * np.log(1.0 + W / R) + const
    ring = MU_S * np.log((R + W) / np.maximum(r, tiny)) + const
    return np.where(r <= R, inside, np.where(r < R + W, ring, const))


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def series(data, key):
    levels = data["levels"]
    h = np.array([1.0 / n for n in levels])
    e = np.array([data["errors"][str(n)][key] for n in levels])
    return h, e


# ---------------------------------------------------------------------------
def fig0_setup(outdir, n_fluid=32):
    """Setup of the benchmark: geometry / BCs, and a zoom on the two meshes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.6))

    # ---- panel (a): domain, structure, boundary conditions -----------------
    ax1.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor="0.88", edgecolor="k",
                                lw=1.2))
    th = np.linspace(0, 2 * np.pi, 500)
    ax1.fill(0.5 + (R + W) * np.cos(th), 0.5 + (R + W) * np.sin(th),
             color="tab:red", alpha=0.85, zorder=2)
    ax1.fill(0.5 + R * np.cos(th), 0.5 + R * np.sin(th), color="0.88",
             zorder=3)
    for rad, ls in ((R, "-"), (R + W, "-")):
        ax1.plot(0.5 + rad * np.cos(th), 0.5 + rad * np.sin(th), "k", ls=ls,
                 lw=0.8, zorder=4)
    ax1.plot([0.5], [0.5], "k+", ms=8, zorder=4)
    box = dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.2)
    ax1.annotate("", xy=(0.5 + R, 0.5), xytext=(0.5, 0.5),
                 arrowprops=dict(arrowstyle="<->", lw=1.0), zorder=5)
    ax1.text(0.5 + R / 2, 0.515, r"$R$", ha="center", va="bottom", fontsize=11,
             zorder=6, bbox=box)
    ax1.annotate("", xy=(0.5 + R + W, 0.5), xytext=(0.5 + R, 0.5),
                 arrowprops=dict(arrowstyle="<->", lw=1.0), zorder=5)
    ax1.text(0.5 + R + W / 2, 0.485, r"$w$", ha="center", va="top", fontsize=11,
             zorder=6, bbox=box)
    ax1.annotate("solid", xy=(0.5 + (R + W / 2) * np.cos(-np.pi / 4),
                              0.5 + (R + W / 2) * np.sin(-np.pi / 4)),
                 xytext=(0.80, 0.20), fontsize=9, color="tab:red", zorder=6,
                 ha="center",
                 arrowprops=dict(arrowstyle="-", lw=0.8, color="tab:red"))
    for x, y, s in ((0.5, 1.035, r"$\mathbf{u}=\mathbf{0}$ (no-slip)"),
                    (0.5, -0.075, r"$\mathbf{u}=\mathbf{0}$")):
        ax1.text(x, y, s, ha="center", fontsize=9)
    ax1.text(-0.075, 0.5, r"$\mathbf{u}=\mathbf{0}$", va="center", rotation=90,
             fontsize=9)
    ax1.text(1.035, 0.5, r"$\mathbf{u}=\mathbf{0}$", va="center", rotation=-90,
             fontsize=9)
    ax1.text(0.06, 0.94, r"fluid $\rho^f,\mu^f$", fontsize=9)
    ax1.text(0.06, 0.10, "pressure datum\n" + r"$p(\mathbf{0})=0$", fontsize=9)
    ax1.set_xlim(-0.16, 1.16)
    ax1.set_ylim(-0.16, 1.14)
    ax1.set_aspect("equal")
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$y$")
    ax1.set_title(r"(a) computational domain $\Omega=[0,1]^2$")

    # ---- panel (b): zoom of the two meshes ---------------------------------
    h = 1.0 / n_fluid
    M = max(n_fluid // 8, 1)
    n_theta = 28 * M
    lo, hi = 0.40, 0.86
    # IB support band: |r-R| < 2h and |r-(R+w)| < 2h  (annuli, not disks)
    for r_out in (R, R + W + 2 * h):
        ax2.add_patch(Annulus((0.5, 0.5), r_out, 2 * h, color="tab:orange",
                              alpha=0.28, ec="none", zorder=1))
    for k in range(int(lo / h) - 1, int(hi / h) + 2):
        ax2.plot([k * h, k * h], [lo, hi], color="0.62", lw=0.5, zorder=2)
        ax2.plot([lo, hi], [k * h, k * h], color="0.62", lw=0.5, zorder=2)
    for j in range(M + 1):                       # solid mesh: radial layers
        rr = R + W * j / M
        ax2.plot(0.5 + rr * np.cos(th), 0.5 + rr * np.sin(th), color="tab:red",
                 lw=0.9, zorder=3)
    for i in range(n_theta):                     # solid mesh: circumferential
        a = 2 * np.pi * i / n_theta
        ax2.plot([0.5 + R * np.cos(a), 0.5 + (R + W) * np.cos(a)],
                 [0.5 + R * np.sin(a), 0.5 + (R + W) * np.sin(a)],
                 color="tab:red", lw=0.6, zorder=3)
    ax2.plot(0.5 + R * np.cos(th), 0.5 + R * np.sin(th), "k-", lw=1.0, zorder=4)
    ax2.plot(0.5 + (R + W) * np.cos(th), 0.5 + (R + W) * np.sin(th), "k-", lw=1.0,
             zorder=4)
    ax2.text(0.645, 0.455, f"Eulerian grid\n$h=1/{n_fluid}$", fontsize=9,
             ha="center", bbox=box, zorder=6)
    ax2.annotate(r"IB support $\pm 2h$", xy=(0.5 + 0.22 * np.cos(0.85),
                                             0.5 + 0.22 * np.sin(0.85)),
                 xytext=(0.53, 0.585), color="darkorange", fontsize=9,
                 ha="center", bbox=box, zorder=6,
                 arrowprops=dict(arrowstyle="-", lw=0.8, color="darkorange"))
    ax2.annotate("solid", xy=(0.5 + 0.28 * np.cos(np.pi / 4),
                              0.5 + 0.28 * np.sin(np.pi / 4)),
                 xytext=(0.82, 0.83), color="tab:red", fontsize=9, ha="center",
                 bbox=box, zorder=6,
                 arrowprops=dict(arrowstyle="-", lw=0.8, color="tab:red"))
    ax2.set_xlim(lo, hi)
    ax2.set_ylim(lo, hi)
    ax2.set_aspect("equal")
    ax2.set_xlabel("$x$")
    ax2.set_ylabel("$y$")
    ax2.set_title(rf"(b) fluid grid + Lagrangian solid mesh ($N={n_fluid}$)")

    fig.tight_layout()
    path = os.path.join(outdir, "annulus_setup.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def fig1_convergence(ipcs, chorin, outdir):
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    h_i, e_p_i = series(ipcs, "e_p_L2")
    _, e_b_i = series(ipcs, "e_p band(|r-R|<2h)")
    _, e_v_i = series(ipcs, "e_v_L2")
    h_c, e_p_c = series(chorin, "e_p_L2")
    _, e_v_c = series(chorin, "e_v_L2")

    hh = np.array([h_i.min() * 0.8, h_i.max() * 1.3])
    for p, style in ((1.0, ":"), (1.5, "--"), (2.0, "-.")):
        ax.plot(hh, e_p_i[0] * (hh / h_i[0]) ** p, style, color="0.65", lw=0.9,
                zorder=1)
        ax.annotate(rf"$h^{{{p:g}}}$", xy=(hh[1], e_p_i[0] * (hh[1] / h_i[0]) ** p),
                    xytext=(3, -1), textcoords="offset points",
                    color="0.4", fontsize=8)

    ax.loglog(h_i, e_p_i, "o-", color="C0", label=r"IPCS  $e_p^{L^2}$ (whole domain)")
    ax.loglog(h_i, e_b_i, "s--", color="C0", mfc="none",
              label=r"IPCS  $e_p^{L^2}$ (IB band $|r-R|<2h$)")
    ax.loglog(h_i, e_v_i, "^-", color="C2", label=r"IPCS  $\|v\|_{L^2}$ (exact $v=0$)")
    ax.loglog(h_c, e_p_c, "o-", color="C3", label=r"Chorin $e_p^{L^2}$ (whole domain)")
    ax.loglog(h_c, e_v_c, "^--", color="C1", mfc="none",
              label=r"Chorin $\|v\|_{L^2}$")

    ax.set_xlabel(r"fluid element size $h = 1/N$")
    ax.set_ylabel("error")
    ax.set_title("demo_423 annular solid: mesh refinement (steady state)")
    ax.grid(True, which="major", alpha=0.3)
    ticks = sorted(set(np.round(np.concatenate([h_i, h_c]), 12)))
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"1/{int(round(1.0 / t))}" for t in ticks])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=2, frameon=False)
    fig.tight_layout()
    path = os.path.join(outdir, "annulus_convergence.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def fig2_pressure_profile(csvs, outdir):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 6.0), sharex=True,
                                   gridspec_kw={"height_ratios": [1.35, 1]})
    ax1.axvspan(R, R + W, color="0.85", zorder=0,
                label="annular solid (fibre direction)")

    finest = None
    ordered = sorted(csvs, key=lambda c: int(os.path.basename(c).split("_N")[-1].split(".")[0]))
    for csv in ordered:
        d = np.loadtxt(csv, delimiter=",", skiprows=1)
        r = d[:, 0] - 0.5
        pn, pex, err = d[:, 1], d[:, 2], d[:, 3]
        m = r <= 0.4
        n = int(os.path.basename(csv).split("_N")[-1].split(".")[0])
        ax1.plot(r[m], pn[m], lw=1.1, label=rf"$N={n}$", zorder=2)
        ax2.semilogy(r[m], np.abs(err[m]), lw=1.0, label=rf"$N={n}$")
        if finest is None or n > finest[0]:
            finest = (n, r[m], pex[m])
    if finest is not None:      # analytical pressure in the same discrete gauge
        ax1.plot(finest[1], finest[2], "k-", lw=1.6, zorder=3,
                 label="analytical $p$")
        ax1.plot(finest[1][::40], finest[2][::40], "kx", ms=3, zorder=3)

    for ax in (ax1, ax2):
        ax.axvspan(R, R + W, color="0.85", zorder=0)
        ax.grid(True, which="both", alpha=0.25)
        ax.set_xlim(0.02, 0.40)
    ax1.set_ylabel(r"$p$")
    ax1.set_title("pressure along the outward radius at $y=0.5$")
    ax1.legend(fontsize=8, loc="lower left")
    ax2.set_xlabel(r"$r$")
    ax2.set_ylabel(r"$|p_h - p_{\rm exact}|$")
    fig.tight_layout()
    path = os.path.join(outdir, "annulus_profile.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def fig3_dt_study(rows, outdir):
    """rows: list of (solver, dt, e_p_L2) at fixed N."""
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for solver, marker, color in (("chorin", "o", "C3"), ("ipcs", "s", "C0")):
        pts = sorted([(dt, e) for s, dt, e in rows if s == solver])
        if not pts:
            continue
        dt = np.array([p[0] for p in pts])
        e = np.array([p[1] for p in pts])
        ax.loglog(dt, e, marker + "-", color=color,
                  label=f"{solver.upper()}  $N=64$")
        if solver == "chorin" and len(pts) >= 2:
            ref = e[0] * (dt / dt[0]) ** 0.7
            ax.loglog(dt, ref, ":", color="0.6", lw=0.9)
            ax.annotate(r"$\propto \Delta t^{0.7}$", xy=(dt[-1], ref[-1]),
                        xytext=(-58, 6), textcoords="offset points",
                        color="0.4", fontsize=8)
    ax.set_xlabel(r"time step $\Delta t$  (final time $T=0.01$ fixed)")
    ax.set_ylabel(r"$e_p^{L^2}$")
    ax.set_title(r"time-step refinement at fixed mesh ($N=64$)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(outdir, "annulus_dt.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def fig4_error_map(xdmf_path, outdir):
    """2-D error map.

    dolfinx 0.10 dropped ``XDMFFile.read_function`` and the file holds two
    grids, which both pyvista and meshio refuse or crash on, so the HDF5
    payload referenced by the XDMF is read directly.
    """
    import h5py
    h5_path = os.path.splitext(xdmf_path)[0] + ".h5"
    with h5py.File(h5_path, "r") as fh:
        coords = np.asarray(fh["Mesh/mesh/geometry"])[:, :2]
        funcs = fh["Function"]
        name = next(n for n in funcs if "error" in n.lower())
        grp = funcs[name]
        vals = np.abs(np.asarray(grp[next(iter(grp.keys()))])[:, 0])
    n = int(round(np.sqrt(coords.shape[0]))) - 1

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    tcf = ax.tricontourf(coords[:, 0], coords[:, 1], vals, levels=40,
                         cmap="inferno")
    th = np.linspace(0, 2 * np.pi, 400)
    for rad in (R, R + W):
        ax.plot(0.5 + rad * np.cos(th), 0.5 + rad * np.sin(th), "c--", lw=0.9)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title(rf"$|p_h - p_{{\rm exact}}|$, IPCS, $N={n}$")
    fig.colorbar(tcf, ax=ax, shrink=0.85)
    fig.tight_layout()
    path = os.path.join(outdir, "annulus_error_map.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ipcs-json", default=None)
    ap.add_argument("--chorin-json", default=None)
    ap.add_argument("--profiles", default=os.path.join(HERE, "plot", "*profile_N*.csv"))
    ap.add_argument("--error-field", default=os.path.join(HERE, "plot", "pressure_error.xdmf"))
    ap.add_argument("--outdir", default=os.path.join(HERE, "figures"))
    ap.add_argument("--n-fluid", type=int, default=32,
                    help="N used for the zoom panel of the setup figure")
    ap.add_argument("--dt-rows", default="chorin:1e-4:3.074286e-3,chorin:2.5e-5:1.018344e-3,"
                                        "chorin:6.25e-6:4.663478e-4,ipcs:1e-4:4.060599e-4,"
                                        "ipcs:2.5e-5:4.060592e-4,ipcs:6.25e-6:4.060593e-4",
                    help="solver:dt:e_p_L2 triples for the dt figure")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    made = []

    made.append(fig0_setup(args.outdir, n_fluid=args.n_fluid))

    if args.ipcs_json and args.chorin_json:
        made.append(fig1_convergence(load_json(args.ipcs_json),
                                     load_json(args.chorin_json), args.outdir))
    csvs = sorted(glob.glob(args.profiles))
    if csvs:
        made.append(fig2_pressure_profile(csvs, args.outdir))

    rows = []
    for item in args.dt_rows.split(","):
        s, dt, e = item.split(":")
        rows.append((s.strip(), float(dt), float(e)))
    made.append(fig3_dt_study(rows, args.outdir))

    if args.error_field and os.path.exists(args.error_field):
        try:
            made.append(fig4_error_map(args.error_field, args.outdir))
        except Exception as exc:      # noqa: BLE001 - figure 4 is optional
            print(f"skipping fig4 ({args.error_field}): {exc}")

    for m in made:
        print("wrote", m)
    if not made:
        print("nothing to do - pass --ipcs-json/--chorin-json and/or --profiles")


if __name__ == "__main__":
    main()
