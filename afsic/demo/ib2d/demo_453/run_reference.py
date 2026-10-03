"""Run the IB2d reference solution of the Single-Porous-Rubberband example.

    python run_reference.py [--tend T] [--out FILE] [--ib2d DIR] [--run DIR]

Pure Python, no MATLAB: the pyIB2d example files (``ib2d_input/``, copied
verbatim from ``third_party/ib2d/pyIB2d/Examples/Single_Porous_Rubberband``
-- they are 0-based, so no index conversion is needed) are copied into a
scratch run directory, pyIB2d's ``IBM_Driver.main`` is executed, and the VTK
output is converted to ``ib2d_reference.npz`` by ``ib2d_reference.py``.

Model switches (identical to the example's ``input2d``): springs + porous
media on, everything else off.
"""
import argparse
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, *[".."] * 4))
sys.path.insert(0, HERE)

import ib2d_io          # noqa: E402
import ib2d_reference   # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tend", type=float, default=None, help="override Tfinal (s)")
    ap.add_argument("--dt", type=float, default=None, help="override dt (s)")
    ap.add_argument("--print-dump", type=int, default=None,
                    help="override print_dump (steps between dumps)")
    ap.add_argument("--src", default=os.path.join(HERE, "ib2d_input"),
                    help="example directory (default: ib2d_input/)")
    ap.add_argument("--out", default=os.path.join(HERE, "ib2d_reference.npz"))
    ap.add_argument("--ib2d", default=os.path.join(REPO, "third_party", "ib2d", "pyIB2d"),
                    help="path of the pyIB2d directory (default third_party)")
    ap.add_argument("--run", default=os.path.join(HERE, "ib2d_run"))
    args = ap.parse_args()

    src = args.src
    P = ib2d_io.read_input2d(os.path.join(src, "input2d"))
    name = P.get("string_name", "rubberband")
    Tfinal = args.tend if args.tend is not None else float(P["Tfinal"])

    if not os.path.isdir(args.ib2d):
        subprocess.run(["git", "clone", "https://github.com/nickabattista/ib2d",
                        os.path.join(REPO, "third_party", "ib2d")], check=True)

    run = args.run
    if os.path.exists(run):
        shutil.rmtree(run)
    os.makedirs(run)
    for ext in ("vertex", "spring", "porous"):
        fname = os.path.join(src, f"{name}.{ext}")
        if os.path.exists(fname):
            shutil.copy(fname, run)
    shutil.copy(os.path.join(src, "input2d"), run)

    blackbox = os.path.join(args.ib2d, "IBM_Blackbox")
    assert os.path.exists(os.path.join(blackbox, "IBM_Driver.py")), \
        f"pyIB2d not found at {args.ib2d}"
    sys.path.insert(0, blackbox)
    sys.path.insert(0, run)
    import IBM_Driver as Driver

    fluid = [float(P["mu"]), float(P["rho"])]
    grid = [int(P["Nx"]), int(P["Ny"]), float(P["Lx"]), float(P["Ly"]),
            int(P["supp"])]
    dt = args.dt if args.dt is not None else float(P["dt"])
    print_dump = (args.print_dump if args.print_dump is not None
                  else int(P["print_dump"]))
    time = [Tfinal, dt]
    lag = [1, 0, 0, 0,               # springs, update_springs, targets, upd
           0, 0, 0, 0,               # beams (invariant), upd_beams, nonInv, upd
           0, 0, 0, 0, 0, 0, 0.0, -1.0,   # muscles, tracers, mass, gravity
           1, 0, 0, 0, 0,            # porous, concentration, electro, damped, upd
           0, 1.0, 0, 0, 0]          # boussinesq, exp_coeff, general, poro, brinkman
    out_par = [print_dump, 0, 0, 0, 0, 0, 0,
               1, 1, 1, 0, 0, 0, 0, 0, 0, 0]

    print(f"[pyIB2d] {name}: Nx={grid[0]} Ny={grid[1]} dt={time[1]:g} "
          f"Tfinal={Tfinal:g} steps={int(Tfinal / time[1])} "
          f"print_dump={out_par[0]}")

    cwd = os.getcwd()
    try:
        os.chdir(run)
        Driver.main(fluid, grid, time, lag, out_par, name)
    finally:
        os.chdir(cwd)

    viz = os.path.join(run, "viz_IB2d")
    assert os.path.isdir(viz), f"pyIB2d wrote no output ({viz} missing)"
    ib2d_reference.main(viz, args.out, dt=time[1], print_dump=out_par[0],
                        nx=grid[0], ny=grid[1])


if __name__ == "__main__":
    main()
