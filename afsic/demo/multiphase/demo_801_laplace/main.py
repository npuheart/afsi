#!/usr/bin/env python3
"""demo_801 - static droplet: the Laplace law ``dp = sigma / R``.

The classic calibration of a two-phase / surface-tension implementation: a
circular droplet of radius ``R`` at rest in a quiescent box with no-slip walls.
At equilibrium the pressure jump across the interface is

    p_in - p_out = sigma * kappa = sigma / R        (2-D, kappa = 1/R) .

Nothing else is going on: there is no inflow, no gravity, no density or
viscosity jump (``rho_1 = rho_2``, ``mu_1 = mu_2``), so the measured jump tests
exactly the surface-tension force, the curvature and the projection together.
The residual velocity is the classical CSF *parasitic current* level.

    python main.py                     # 64x64, eps = h, writes results.csv
    python main.py --nx 128 --eps 2.0  # mesh refinement / interface-width study
    python main.py --sweep             # full table over (nx, eps) pairs

Outputs (in ``output/``): ``laplace.csv`` with one row per run and, with
``--plot``, a pressure profile figure through the droplet centre.
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
    p.add_argument("--nx", type=int, default=64, help="grid cells per direction")
    p.add_argument("--ny", type=int, default=None, help="defaults to --nx")
    p.add_argument("--R", type=float, default=0.25, help="droplet radius")
    p.add_argument("--sigma", type=float, default=1.0, help="surface tension")
    p.add_argument("--eps", type=float, default=1.0,
                   help="interface parameter in units of the mesh size h")
    p.add_argument("--mu", type=float, default=0.05, help="viscosity (both fluids)")
    p.add_argument("--rho", type=float, default=1.0, help="density (both fluids)")
    p.add_argument("--dt", type=float, default=None,
                   help="time step (default: 0.3 * capillary limit)")
    p.add_argument("--tend", type=float, default=None,
                   help="end time (default: 0.08, i.e. until the jump settles)")
    p.add_argument("--mobility", type=float, default=0.05, help="Allen-Cahn mobility")
    p.add_argument("--smooth-passes", type=int, default=2)
    p.add_argument("--fst-smooth", type=int, default=1)
    p.add_argument("--sweep", action="store_true",
                   help="run the built-in (nx, eps) convergence table")
    p.add_argument("--plot", action="store_true", help="write a pressure-profile figure")
    return p.parse_args()


def capillary_dt(sigma, rho, h):
    """Brackbill et al. capillary time-step limit ``sqrt(rho h^3 / (2 pi sigma))``."""
    return np.sqrt(rho * h ** 3 / (2.0 * np.pi * sigma)) if sigma > 0 else np.inf


def run_case(nx, ny, R, sigma, eps_factor, mu, rho, dt, tend, mobility,
             smooth_passes, fst_smooth, verbose=True):
    L = 1.0
    eps = eps_factor * L / nx
    h = L / nx
    if dt is None:
        dt = 0.3 * capillary_dt(sigma, rho, h)
    if tend is None:
        tend = 0.08
    centre = (0.5 * L, 0.5 * L)
    solver = TwoPhaseSolver((0.0, L, 0.0, L), nx, ny, dt,
                            rho1=rho, rho2=rho, mu1=mu, mu2=mu,
                            sigma=sigma, eps=eps, mobility=mobility,
                            smooth_passes=smooth_passes, fst_smooth=fst_smooth,
                            bc_spec=lambda x: np.zeros((2, x.shape[1])))
    solver.set_phase(solver.droplet_phase(centre, R, eps))
    solver.set_velocity()
    t0 = time.time()
    solver.run(tend)
    wall = time.time() - t0

    p_in, p_out = solver.interior_pressure(R, centre)
    jump = p_in - p_out
    exact = sigma / R
    a_eq, major, minor, _ = solver.droplet_axes(centre=centre, window=0.45)
    row = dict(nx=nx, ny=ny, eps_over_h=eps_factor, eps=eps, dt=dt, tend=tend,
               sigma=sigma, R=R, mu=mu, rho=rho,
               dp=jump, dp_exact=exact, rel_error=abs(jump - exact) / exact,
               a_eq=a_eq, radius_rel_error=abs(a_eq - R) / R,
               D=(major - minor) / (major + minor),
               u_parasitic=solver.spurious_velocity(),
               ca_parasitic=mu * solver.spurious_velocity() / sigma if sigma else 0.0,
               steps=solver.steps, wall_s=wall)
    if verbose:
        print(f"nx={nx:4d} eps={eps_factor:4.2f}h dt={dt:.3e}  "
              f"dp={jump:+.5f} (exact {exact:.5f}) relerr={100*row['rel_error']:+.3f}%  "
              f"a_eq={a_eq:.5f}  |u|={row['u_parasitic']:.3e}  "
              f"D={row['D']:+.1e}  {wall:.1f}s")
    return row, solver


def main():
    args = parse_args()
    os.makedirs(OUT, exist_ok=True)
    ny = args.ny or args.nx

    if args.sweep:
        cases = [(32, 1.0), (64, 1.0), (128, 1.0), (64, 2.0), (128, 2.0), (64, 0.75)]
    else:
        cases = [(args.nx, args.eps)]

    rows, last = [], None
    for nx, ef in cases:
        row, last = run_case(nx, ny if nx == args.nx else nx, args.R, args.sigma,
                             ef, args.mu, args.rho, args.dt, args.tend,
                             args.mobility, args.smooth_passes, args.fst_smooth)
        rows.append(row)

    path = os.path.join(OUT, "laplace.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {path}")
    if len(rows) > 1:
        print("\nsummary (relative error of the Laplace jump):")
        for r in rows:
            print(f"  nx={r['nx']:4d} eps={r['eps_over_h']:.2f}h -> "
                  f"{100*r['rel_error']:+.3f}%   parasitic Ca={r['ca_parasitic']:.2e}")

    if args.plot:
        plot_profile(last, args.R)


def plot_profile(solver, R):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available - skipping figure")
        return
    L = solver.box[1]
    X = solver.Q.tabulate_dof_coordinates()[:, :2]
    sel = np.isclose(X[:, 1], 0.5 * L, atol=0.5 * solver.hy)
    order = np.argsort(X[sel, 0])
    x = X[sel, 0][order]
    p = solver.p_.x.array[: X.shape[0]][sel][order]
    phi = solver.phi.x.array[: X.shape[0]][sel][order]
    fig, ax = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    ax[0].plot(x, p, "o-", ms=3)
    ax[0].axvline(0.5 * L - R, color="k", ls=":", lw=1)
    ax[0].axvline(0.5 * L + R, color="k", ls=":", lw=1)
    ax[0].set_ylabel("pressure p")
    ax[0].set_title("demo_801: static droplet (dotted = exact interface radius)")
    ax[1].plot(x, phi)
    ax[1].set_ylabel(r"phase $\phi$")
    ax[1].set_xlabel("x")
    fig.tight_layout()
    fig_path = os.path.join(OUT, "laplace_profile.png")
    fig.savefig(fig_path, dpi=140)
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
