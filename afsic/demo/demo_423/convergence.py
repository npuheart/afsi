#!/usr/bin/env python
"""Mesh-refinement (convergence-order) study for demo_423.

The demo solves the static equilibrium of an immersed anisotropic annular
solid on a fixed Cartesian fluid mesh (N x N quadrilaterals) coupled to a
Lagrangian solid mesh (M = N/8 radial layers, 28*M elements around the
circumference).  Both meshes are refined together, so halving the fluid
element size h = 1/N also halves the solid element size.

This script runs ``generate_mesh.py`` and ``main.py`` for a sequence of
levels and reports the observed convergence order

    p = log(e_N / e_2N) / log 2

for every error measure printed by ``main.py``.

Usage (inside the afsi-dolfinx environment, from this directory):

    python convergence.py                       # N = 16 32 64 128
    python convergence.py -n 16 32 64 --steps 200
    LEVELS=32,64 STEPS=100 python convergence.py
    python convergence.py --dt 1e-5 --steps 1000 --json results.json

Environment variables LEVELS / STEPS / DT / SOLVER / CELL_TYPE are honoured
so the script can also be driven from a shell loop.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))

# Error measures printed by main.py, in reporting order.
METRICS = [
    ("e_p_L2", "pressure L2 (whole domain)"),
    ("e_p inner(r<R-2h)", "pressure L2, inner disk r<R-2h"),
    ("e_p inner4(r<R-4h)", "pressure L2, inner disk r<R-4h"),
    ("e_p band(|r-R|<2h)", "pressure L2, IB band |r-R|<2h"),
    ("e_p outer(r>R+w+2h)", "pressure L2, outer r>R+w+2h"),
    ("e_v_L2", "velocity L2 (exact v=0)"),
    ("e_v_H1", "velocity H1 (exact v=0)"),
    ("max|v|", "max |v|"),
]

_NUM = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")


def parse_output(text: str) -> dict[str, float]:
    """Collect ``key = value`` pairs printed by main.py."""
    values: dict[str, float] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, rest = line.split("=", 1)
        key = key.strip()
        token = rest.strip().split(",")[0].strip().split(" ")[0]
        if _NUM.match(token):
            values.setdefault(key, float(token))
    return values


def run_level(n: int, steps: int, dt: str, solver: str, cell_type: str,
              verbose: bool) -> tuple[dict[str, float], str, float]:
    env = dict(os.environ)
    env.update(N=str(n), STEPS=str(steps), DT=str(dt), SOLVER=solver,
               CELL_TYPE=cell_type)
    t0 = time.time()

    gen = subprocess.run([sys.executable, "generate_mesh.py"], cwd=DEMO_DIR,
                         env=env, capture_output=True, text=True)
    if gen.returncode != 0:
        sys.stderr.write(gen.stdout + gen.stderr)
        raise SystemExit(f"generate_mesh.py failed for N={n}")
    if verbose:
        print(gen.stdout.strip())

    run = subprocess.run([sys.executable, "main.py"], cwd=DEMO_DIR, env=env,
                         capture_output=True, text=True)
    elapsed = time.time() - t0
    if run.returncode != 0:
        sys.stderr.write(run.stdout[-4000:] + run.stderr[-4000:])
        raise SystemExit(f"main.py failed for N={n}")

    if verbose:
        print(run.stdout.strip())
    return parse_output(run.stdout), run.stdout, elapsed


def observed_order(e_coarse: float, e_fine: float, ratio: float) -> float:
    if e_coarse <= 0.0 or e_fine <= 0.0:
        return float("nan")
    return math.log(e_coarse / e_fine) / math.log(ratio)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    default_levels = [int(x) for x in
                      os.environ.get("LEVELS", "16,32,64,128").split(",")]
    ap.add_argument("-n", "--levels", type=int, nargs="+", default=default_levels,
                    help="fluid mesh sizes N (default: 16 32 64 128)")
    ap.add_argument("--steps", type=int,
                    default=int(os.environ.get("STEPS", "100")))
    ap.add_argument("--dt", default=os.environ.get("DT", "1.0e-4"))
    ap.add_argument("--solver", default=os.environ.get("SOLVER", "chorin"))
    ap.add_argument("--cell-type", default=os.environ.get("CELL_TYPE", "quadrilateral"))
    ap.add_argument("--json", default=None, help="write raw results to this JSON file")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    levels = sorted(set(args.levels))
    if len(levels) < 2:
        ap.error("need at least two refinement levels to measure an order")

    print(f"demo_423 convergence study: N={levels}, steps={args.steps}, "
          f"dt={args.dt}, T={args.steps * float(args.dt):g}, "
          f"solver={args.solver}, cells={args.cell_type}")
    print(f"(solid mesh: M=N/8 = {[max(n // 8, 1) for n in levels]} radial layers)\n")

    results: dict[int, dict[str, float]] = {}
    raw: dict[int, str] = {}
    timings: dict[int, float] = {}
    for n in levels:
        print(f"--- running N={n} ---", flush=True)
        values, stdout, elapsed = run_level(n, args.steps, args.dt, args.solver,
                                            args.cell_type, args.verbose)
        results[n] = values
        raw[n] = stdout
        timings[n] = elapsed
        p = values.get("e_p_L2")
        print(f"    done in {elapsed:6.1f} s   e_p_L2 = {p:.6e}" if p is not None
              else f"    done in {elapsed:6.1f} s")

    # ---- error table ------------------------------------------------------
    present = [(k, lbl) for k, lbl in METRICS
               if any(k in results[n] for n in levels)]
    width = max(len(lbl) for _, lbl in present)
    header = f"{'error measure':<{width}} | " + " | ".join(f"N={n:<10d}" for n in levels)
    print("\n" + "=" * len(header))
    print("Errors")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for key, label in present:
        row = " | ".join(f"{results[n].get(key, float('nan')):.6e}" for n in levels)
        print(f"{label:<{width}} | {row}")

    # ---- order table ------------------------------------------------------
    print("\n" + "=" * 90)
    print("Observed convergence order  p = log(e_N / e_2N) / log(N_2N / N)")
    print("=" * 90)
    pairs = list(zip(levels[:-1], levels[1:]))
    owidth = max(len(lbl) for _, lbl in present)
    header2 = f"{'error measure':<{owidth}} | " + " | ".join(
        f"{a}->{b:<8d}" for a, b in pairs)
    print(header2)
    print("-" * len(header2))
    orders: dict[str, list[float | None]] = {}
    for key, label in present:
        row_orders = []
        for a, b in pairs:
            ea, eb = results[a].get(key), results[b].get(key)
            if ea is None or eb is None:
                row_orders.append(None)
                continue
            row_orders.append(observed_order(ea, eb, b / a))
        orders[key] = row_orders
        cells = " | ".join(
            "     n/a " if p is None else f"{p:8.3f}" for p in row_orders)
        print(f"{label:<{owidth}} | {cells}")
    print("=" * 90)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({
                "levels": levels,
                "steps": args.steps,
                "dt": args.dt,
                "solver": args.solver,
                "cell_type": args.cell_type,
                "errors": {str(n): results[n] for n in levels},
                "orders": {k: v for k, v in orders.items()},
                "seconds": {str(n): timings[n] for n in levels},
                "stdout": {str(n): raw[n] for n in levels},
            }, fh, indent=2)
        print(f"\nRaw results written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
