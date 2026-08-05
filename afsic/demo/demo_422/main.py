"""demo_422 command-line entry point.

Monolithic IBFE: lid-driven square cavity with an immersed elastic disk.
"""
import argparse

from config import make_config
from immersed import ImmersedFEM


def main():
    p = argparse.ArgumentParser(
        description="Monolithic IBFE: lid-driven cavity + immersed elastic disk")
    p.add_argument("--steps", type=int, help="number of time steps")
    p.add_argument("--nx", "--ny", dest="nxy", type=int, help="background mesh size")
    p.add_argument("--scheme", type=int, choices=[0, 3], help="0 monolithic | 3 reduced")
    p.add_argument("--dt", type=float, help="time step")
    p.add_argument("--mu-s", type=float, dest="mu_s", help="solid shear modulus")
    p.add_argument("--out", type=int, help="output interval (steps)")
    args = p.parse_args()

    cfg = make_config()
    if args.steps is not None:
        cfg["num_steps"] = args.steps
    if args.nxy is not None:
        cfg["Nx"] = cfg["Ny"] = args.nxy
    if args.scheme is not None:
        cfg["scheme"] = args.scheme
    if args.dt is not None:
        cfg["dt"] = args.dt
    if args.mu_s is not None:
        cfg["mu_s"] = args.mu_s
    if args.out is not None:
        cfg["out_every"] = args.out

    problem = ImmersedFEM(cfg)
    problem.run()


if __name__ == "__main__":
    main()
