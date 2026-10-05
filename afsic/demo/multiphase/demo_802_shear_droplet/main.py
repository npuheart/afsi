#!/usr/bin/env python3
"""demo_802 - droplet in simple shear: Taylor deformation.

A circular droplet of radius ``a`` is placed at the centre of a box in which an
exact linear shear ``u = (gdot (y - L/2), 0)`` is imposed on **all** walls.  That
profile is an exact steady solution of the Navier-Stokes equations, the velocity
vanishes at the droplet centre so the droplet stays put, and the interface
reaches a stationary tilted ellipse.  For small capillary number

    Ca = mu_matrix * gdot * a / sigma

Taylor's small-deformation theory gives

    D = (L - B) / (L + B) = Ca (19 lambda + 16) / (16 lambda + 16) ,
    lambda = mu_drop / mu_matrix ,        tilt angle = 45 degrees

with ``L``/``B`` the major/minor axis lengths (from the second moments of the
``phi >= 0.5`` region).  The viscosity jump ``lambda != 1`` is what exercises the
variable-coefficient part of the solver (``mu(phi)`` re-assembled every step).

    python main.py                 # Ca = 0.05, lambda = 2
    python main.py --sweep         # Ca = 0.025 / 0.05 / 0.1 table
    python main.py --lam 1         # same-viscosity droplet

Outputs ``output/taylor.csv`` (+ ``output/taylor.png`` with ``--plot``).
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
    p.add_argument("--R", type=float, default=0.15, help="droplet radius a")
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--mu", type=float, default=0.05, help="matrix viscosity")
    p.add_argument("--lam", type=float, default=2.0, help="viscosity ratio mu_drop/mu_matrix")
    p.add_argument("--rho-ratio", type=float, default=1.0, help="rho_drop/rho_matrix")
    p.add_argument("--Ca", type=float, default=0.05)
    p.add_argument("--eps", type=float, default=1.0, help="interface width in cells")
    p.add_argument("--strain", type=float, default=1.0, help="total strain gdot*tend")
    p.add_argument("--mobility", type=float, default=0.005,
                   help="Allen-Cahn mobility (small: the AC equation is not "
                        "area preserving and a droplet drifts at rate ~ M eps / R)")
    p.add_argument("--dt", type=float, default=3e-4)
    p.add_argument("--sweep", action="store_true", help="run a Ca sweep")
    p.add_argument("--plot", action="store_true")
    return p.parse_args()


def taylor_deformation(Ca, lam):
    """Taylor (1932) small-Ca steady deformation."""
    return Ca * (19.0 * lam + 16.0) / (16.0 * lam + 16.0)


def run_case(nx, ny, R, sigma, mu_m, lam, rho_ratio, Ca, eps_factor, strain,
             mobility, dt, verbose=True):
    L = 1.0
    eps = eps_factor * L / nx
    gdot = Ca * sigma / (mu_m * R)
    solver = TwoPhaseSolver((0.0, L, 0.0, L), nx, ny, dt,
                            rho1=1.0, rho2=rho_ratio,
                            mu1=mu_m, mu2=lam * mu_m,
                            sigma=sigma, eps=eps, mobility=mobility,
                            bc_spec=lambda x: np.array([gdot * (x[1] - 0.5 * L),
                                                        np.zeros_like(x[0])]))
    centre = (0.5 * L, 0.5 * L)
    solver.set_phase(solver.droplet_phase(centre, R, eps))
    solver.set_velocity(ux=lambda x: gdot * (x[1] - 0.5 * L))
    t0 = time.time()
    solver.run(strain / gdot)
    wall = time.time() - t0

    a_eq, major, minor, theta = solver.droplet_axes(centre=centre, window=0.4)
    D = (major - minor) / (major + minor)
    D_taylor = taylor_deformation(Ca, lam)
    row = dict(nx=nx, Ca=Ca, lam=lam, rho_ratio=rho_ratio, sigma=sigma, mu_matrix=mu_m,
               R=R, eps_over_h=eps_factor, eps=eps, gdot=gdot, dt=dt, strain=strain,
               a_eq=a_eq, area_ratio=(a_eq / R) ** 2,
               D=D, D_taylor=D_taylor, ratio=D / D_taylor if D_taylor else 0.0,
               tilt_deg=np.degrees(theta) % 180.0,
               u_max=solver.spurious_velocity(), steps=solver.steps, wall_s=wall)
    if verbose:
        print(f"nx={nx:4d} Ca={Ca:.3f} lam={lam:g} rho2/rho1={rho_ratio:g}  "
              f"D={D:+.5f} Taylor={D_taylor:.5f} ratio={row['ratio']:.3f}  "
              f"a_eq/R={a_eq/R:.4f} tilt={row['tilt_deg']:.1f}deg  {wall:.0f}s")
    return row, solver


def main():
    args = parse_args()
    os.makedirs(OUT, exist_ok=True)
    ny = args.ny or args.nx
    cases = [0.025, 0.05, 0.1] if args.sweep else [args.Ca]
    rows, last = [], None
    for Ca in cases:
        row, last = run_case(args.nx, ny, args.R, args.sigma, args.mu, args.lam,
                             args.rho_ratio, Ca, args.eps, args.strain,
                             args.mobility, args.dt)
        rows.append(row)
    path = os.path.join(OUT, "taylor.csv")
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
            Ca = np.array([r["Ca"] for r in rows])
            ax.plot(Ca, [r["D_taylor"] for r in rows], "k--", label="Taylor")
            ax.plot(Ca, [r["D"] for r in rows], "o-", label="measured")
            ax.set_xlabel("Ca"); ax.set_ylabel("D"); ax.legend()
            fig.savefig(os.path.join(OUT, "taylor.png"), dpi=140)
            print("wrote output/taylor.png")
        except ImportError:
            print("matplotlib not available - skipping figure")


if __name__ == "__main__":
    main()
