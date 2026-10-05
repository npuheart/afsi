#!/usr/bin/env python3
"""demo_804 - Rayleigh-Taylor instability (density jump + gravity).

A heavy fluid (``rho_2``) rests on a light one (``rho_1``) in a closed box with
no-slip walls; the interface is perturbed by a single cosine mode of amplitude
``A_0`` and wavelength ``L``.  This is the classical Rayleigh-Taylor
configuration and it exercises the *variable-density* part of the solver
(``rho(phi)`` re-assembled every step) together with the buoyancy force
``(rho - rho_1) g``.

For the inviscid problem the linear growth rate is ``sqrt(A_t g k)`` with the
Atwood number ``A_t = (rho_2 - rho_1)/(rho_2 + rho_1)`` and ``k = 2 pi / L``;
with matched viscosities the viscous correction

    sigma^2 + 2 nu k^2 sigma = A_t g k ,     nu = (mu/rho_1 + mu/rho_2)/2

is what the discrete growth rate should follow.  Because the interface is
diffuse (thickness ``4 eps``) and the amplitude becomes finite, the measured rate
is reported as an order-of-magnitude check, not as a sharp-interface benchmark.

    python main.py                 # 64x64, rho2/rho1 = 3, g = 1
    python main.py --sweep-mu      # mu = 0.02 / 0.01 / 0.005 table
    python main.py --plot

Outputs ``output/rayleigh_taylor.csv`` (+ ``.png``).
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
    p.add_argument("--ratio", type=float, default=3.0, help="rho_2 / rho_1")
    p.add_argument("--g", type=float, default=1.0)
    p.add_argument("--mu", type=float, default=0.005)
    p.add_argument("--A0", type=float, default=0.05, help="initial amplitude")
    p.add_argument("--eps", type=float, default=1.5, help="interface width in cells")
    p.add_argument("--mobility", type=float, default=0.05)
    p.add_argument("--tend", type=float, default=0.6)
    p.add_argument("--dt", type=float, default=5e-4)
    p.add_argument("--sweep-mu", action="store_true")
    p.add_argument("--plot", action="store_true")
    return p.parse_args()


def theory_rate(ratio, g, mu, L, rhos=(1.0,)):
    At = (ratio - 1.0) / (ratio + 1.0)
    k = 2.0 * np.pi / L
    inv = np.sqrt(At * g * k)
    nu = 0.5 * (mu / 1.0 + mu / ratio)
    s = (-2 * nu * k ** 2 + np.sqrt((2 * nu * k ** 2) ** 2 + 4 * At * g * k)) / 2.0
    return At, k, inv, nu, s


def run_case(nx, ny, ratio, g, mu, A0, eps_factor, mobility, tend, dt, verbose=True):
    L = 1.0
    eps = eps_factor * L / nx
    rho1, rho2 = 1.0, ratio
    solver = TwoPhaseSolver((0.0, L, 0.0, L), nx, ny, dt,
                            rho1=rho1, rho2=rho2, mu1=mu, mu2=mu,
                            sigma=0.0, eps=eps, mobility=mobility,
                            gravity=(0.0, -g),
                            bc_spec=lambda x: np.zeros((2, x.shape[1])))
    centre = 0.5 * L
    # phi = 1 (heavy) for y > 0.5, perturbed by A0 cos(2 pi x / L)
    solver.set_phase(lambda x: 0.5 * (1 + np.tanh(
        (x[1] - centre - A0 * np.cos(2 * np.pi * x[0] / L)) / (2 * np.sqrt(2) * eps))))
    solver.set_velocity()

    ts, amps, umax = [], [], []
    t0 = time.time()
    n = nx
    while solver._t < tend - 1e-12:
        solver.solve_one_step()
        if solver.steps % 20 == 0:
            # phi increases from the light fluid (0) to the heavy one (1) with y,
            # so the 0.5 crossing is a direct interpolation on the column
            col = solver.nodal(solver.phi)[:, n // 2]
            ys = np.linspace(0.0, L, n + 1)
            y_int = float(np.interp(0.5, col, ys))
            ts.append(solver._t)
            amps.append(abs(y_int - centre))
            umax.append(float(np.abs(solver.u_n.x.array).max()))
    wall = time.time() - t0
    ts, amps = np.array(ts), np.array(amps)
    # local growth rate over the exponential window
    sel = amps > 1.05 * A0
    if sel.sum() > 5:
        fit = np.polyfit(ts[sel], np.log(amps[sel]), 1)
        rate = float(fit[0])
    else:
        rate = float("nan")
    At, k, inv, nu, visc = theory_rate(ratio, g, mu, L)
    row = dict(nx=nx, rho_ratio=ratio, g=g, mu=mu, A0=A0, eps_over_h=eps_factor, eps=eps,
               mobility=mobility, dt=dt, tend=tend,
               Atwood=At, k=k, rate_inviscid=inv, nu_mean=nu, rate_viscous_theory=visc,
               rate_measured=rate, ratio_to_inviscid=rate / inv,
               ratio_to_viscous=rate / visc, amp_end=float(amps[-1]),
               u_max_end=float(umax[-1]), steps=solver.steps, wall_s=wall)
    if verbose:
        print(f"nx={nx:4d} rho2/rho1={ratio:g} mu={mu:g}  meas={rate:.3f} "
              f"inviscid={inv:.3f} viscous={visc:.3f}  ratio(visc)={row['ratio_to_viscous']:.2f}  "
              f"amp {A0}->{amps[-1]:.3f}  {wall:.0f}s")
    return row, ts, amps, inv, visc


def main():
    args = parse_args()
    os.makedirs(OUT, exist_ok=True)
    ny = args.ny or args.nx
    mus = [0.02, 0.01, 0.005] if args.sweep_mu else [args.mu]
    rows, curves = [], []
    for mu in mus:
        row, ts, amps, inv, visc = run_case(args.nx, ny, args.ratio, args.g, mu, args.A0,
                                            args.eps, args.mobility, args.tend, args.dt)
        rows.append(row)
        curves.append((mu, ts, amps, inv, visc))
    path = os.path.join(OUT, "rayleigh_taylor.csv")
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
            for mu, ts, amps, inv, visc in curves:
                ax.semilogy(ts, amps, ".", ms=4, label=rf"$\mu$={mu:g} (measured)")
                ax.semilogy(ts, args.A0 * np.exp(visc * ts), "k--", lw=1)
            ax.set_xlabel("t"); ax.set_ylabel("interface amplitude")
            ax.set_title("demo_804: Rayleigh-Taylor (dashed = viscous linear theory)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(OUT, "rayleigh_taylor.png"), dpi=140)
            print("wrote output/rayleigh_taylor.png")
        except ImportError:
            print("matplotlib not available - skipping figure")


if __name__ == "__main__":
    main()
