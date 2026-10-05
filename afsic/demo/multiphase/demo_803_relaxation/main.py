#!/usr/bin/env python3
"""demo_803 - relaxation of a deformed droplet (Maffettone-Minale time).

A droplet that is released from a slightly elliptical shape relaxes back to a
circle.  Linear theory for a viscous drop in a viscous matrix (Taylor 1932;
Cox 1969; Maffettone & Minale 1998) gives an exponential decay
``D(t) = D_0 exp(-t / tau)`` with

    tau = (R mu_matrix / sigma) * (2 lambda + 3)(19 lambda + 16) / (20 (lambda + 1))

with ``lambda = mu_drop / mu_matrix`` and ``R`` the equivalent radius.  Because
the drive is the interface's own curvature (not an imposed flow), this test has
no ambient shear and no confinement coupling to the flow field: it isolates the
*balance* between the surface tension force and viscous dissipation, which is
exactly what the two-phase solver has to get right.

    python main.py              # R=0.25, lambda=1, N=64
    python main.py --lam 2      # viscosity jump
    python main.py --sweep      # lambda = 0.5 / 1 / 2 table

Outputs ``output/relaxation.csv`` and (with ``--plot``) ``output/relaxation.png``.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from two_phase import TwoPhaseSolver                                     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--nx", type=int, default=64)
    p.add_argument("--ny", type=int, default=None)
    p.add_argument("--R", type=float, default=0.25)
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--mu", type=float, default=0.05, help="matrix viscosity")
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--eps", type=float, default=1.0, help="interface width in cells")
    p.add_argument("--mobility", type=float, default=0.005,
                   help="small: the Allen-Cahn equation is not area preserving")
    p.add_argument("--D0", type=float, default=0.1, help="initial deformation")
    p.add_argument("--dt", type=float, default=2e-4)
    p.add_argument("--tend", type=float, default=None, help="default: 1.8*tau")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--plot", action="store_true")
    return p.parse_args()


def maffettone_minale_tau(R, mu_m, sigma, lam):
    return (R * mu_m / sigma) * (2.0 * lam + 3.0) * (19.0 * lam + 16.0) / (20.0 * (lam + 1.0))


def run_case(nx, ny, R, sigma, mu_m, lam, eps_factor, mobility, D0, dt, tend, verbose=True):
    L = 1.0
    eps = eps_factor * L / nx
    tau = maffettone_minale_tau(R, mu_m, sigma, lam)
    if tend is None:
        tend = 1.8 * tau
    solver = TwoPhaseSolver((0.0, L, 0.0, L), nx, ny, dt,
                            rho1=1.0, rho2=1.0, mu1=mu_m, mu2=lam * mu_m,
                            sigma=sigma, eps=eps, mobility=mobility,
                            bc_spec=lambda x: np.zeros((2, x.shape[1])))
    centre = (0.5 * L, 0.5 * L)
    a = R * np.sqrt((1 + D0) / (1 - D0))
    b = R * np.sqrt((1 - D0) / (1 + D0))

    def ellipse(x):
        dx, dy = x[0] - centre[0], x[1] - centre[1]
        g = np.sqrt((dx / a) ** 2 + (dy / b) ** 2) - 1.0
        r = np.hypot(dx, dy)
        d = np.where(r > 0, g * r, -b)
        return 0.5 * (1 - np.tanh(d / (2 * np.sqrt(2) * eps)))

    solver.set_phase(ellipse)
    solver.set_velocity()

    ts, Ds = [], []
    t0 = time.time()
    while solver._t < tend - 1e-12:
        solver.solve_one_step()
        if solver.steps % 20 == 0:
            _, major, minor, _ = solver.droplet_axes(centre=centre, window=0.45)
            ts.append(solver._t)
            Ds.append((major - minor) / (major + minor))
    wall = time.time() - t0
    ts, Ds = np.array(ts), np.array(Ds)
    sel = (Ds > 0.02 * D0) & (Ds > 0.002)
    fit = np.polyfit(ts[sel], np.log(np.abs(Ds[sel])), 1)
    tau_meas = -1.0 / fit[0]
    row = dict(nx=nx, R=R, lam=lam, sigma=sigma, mu_matrix=mu_m, eps_over_h=eps_factor,
               eps=eps, mobility=mobility, D0=D0, dt=dt, tend=tend,
               tau_theory=tau, tau_measured=tau_meas, ratio=tau_meas / tau,
               D_end=float(Ds[-1]), steps=solver.steps, wall_s=wall)
    if verbose:
        print(f"nx={nx:4d} lam={lam:g} eps={eps_factor:.2f}h  tau_theory={tau:.4f} "
              f"tau_measured={tau_meas:.4f} ratio={row['ratio']:.3f}  "
              f"D({tend:.2f})={Ds[-1]:.4f}  {wall:.0f}s")
    return row, ts, Ds, tau


def main():
    args = parse_args()
    os.makedirs(OUT, exist_ok=True)
    ny = args.ny or args.nx
    lams = [0.5, 1.0, 2.0] if args.sweep else [args.lam]
    rows, curves = [], []
    for lam in lams:
        row, ts, Ds, tau = run_case(args.nx, ny, args.R, args.sigma, args.mu, lam,
                                    args.eps, args.mobility, args.D0, args.dt, args.tend)
        rows.append(row)
        curves.append((lam, ts, Ds, tau))
    path = os.path.join(OUT, "relaxation.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {path}")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            for lam, ts, Ds, tau in curves:
                ax.semilogy(ts, Ds, ".", ms=4, label=rf"$\lambda$={lam:g}")
            ts = curves[0][1]
            ax.semilogy(ts, args.D0 * np.exp(-ts / curves[0][3]), "k--", lw=1,
                        label="analytic")
            ax.set_xlabel("t"); ax.set_ylabel("D")
            ax.set_title("demo_803: droplet relaxation")
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(OUT, "relaxation.png"), dpi=140)
            print("wrote output/relaxation.png")
        except ImportError:
            print("matplotlib not available - skipping figure")


if __name__ == "__main__":
    main()
