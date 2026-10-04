"""Run the IB2d reference solution of the moving-circle example.

    python run_reference.py [--tend T] [--out FILE] [--ib2d DIR] [--run DIR]

Pure Python, no MATLAB.  The example is one of the ``matIB2d`` ones (1-based
identifiers), so the files are copied verbatim from ``ib2d_input/`` but a rebased
copy of ``.target`` is written into the scratch run directory, together with the
three state files and the python port of the example's
``update_Target_Point_Positions.m`` (which pyIB2d imports from there and which
selects the interpolation scheme with the ``INTERP`` environment variable, so
one port serves both the Linear_Interp and the Cubic_Interp variants).

The VTK output of pyIB2d (fluid + structure) is converted to
``ib2d_reference.npz`` by ``ib2d_reference.py``.
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


def shift_target(path_in, path_out, index_base=1):
    """Copy a ``.target`` file with its identifiers rebased to 0."""
    with open(path_in) as fh:
        rows = np.atleast_2d(np.loadtxt(fh, skiprows=1))
    rows[:, 0] -= index_base
    with open(path_out, "w") as fh:
        fh.write(f"{rows.shape[0]}\n")
        for r in rows:
            fh.write(f"{int(r[0])} {r[1]:1.16e}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tend", type=float, default=None, help="override Tfinal (s)")
    ap.add_argument("--dt", type=float, default=None, help="override dt (s)")
    ap.add_argument("--print-dump", type=int, default=None,
                    help="override print_dump (steps between dumps)")
    ap.add_argument("--src", default=os.path.join(HERE, "ib2d_input"),
                    help="example directory (default: ib2d_input/)")
    ap.add_argument("--out", default=os.path.join(HERE, "ib2d_reference.npz"))
    ap.add_argument("--ib2d", default=os.path.join(REPO, "third_party", "ib2d",
                                                   "pyIB2d"),
                    help="path of the pyIB2d directory (default third_party)")
    ap.add_argument("--run", default=os.path.join(HERE, "ib2d_run"))
    args = ap.parse_args()

    src = args.src
    P = ib2d_io.read_input2d(os.path.join(src, "input2d"))
    name = P.get("string_name", "rubberband")
    Tfinal = (args.tend if args.tend is not None
              else float(os.environ.get("TFINAL", P["Tfinal"])))

    if not os.path.isdir(args.ib2d):
        subprocess.run(["git", "clone", "https://github.com/nickabattista/ib2d",
                        os.path.join(REPO, "third_party", "ib2d")], check=True)

    run = args.run
    if os.path.exists(run):
        shutil.rmtree(run)
    os.makedirs(run)
    shutil.copy(os.path.join(src, f"{name}.vertex"), run)
    shift_target(os.path.join(src, f"{name}.target"),
                 os.path.join(run, f"{name}.target"))
    shutil.copy(os.path.join(src, "input2d"), run)
    # the driver imports update_Target_Point_Positions from the run directory
    shutil.copy(os.path.join(src, "update_Target_Point_Positions.py"), run)
    for nm in "ABC":
        shutil.copy(os.path.join(src, f"State_{nm}.pts"), run)

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
                  else int(os.environ.get("PRINT_DUMP", P["print_dump"])))
    time = [Tfinal, dt]
    lag = [0, 0, 1, 1,               # springs, update_springs, targets, upd
           0, 0, 0, 0,               # beams, upd_beams, nonInv, upd
           0, 0, 0, 0, 0, 0, 0.0, -1.0,   # muscles, tracers, mass, gravity
           0, 0, 0, 0, 0,           # porous, concentration, electro, damped, upd
           0, 1.0, 0, 0, 0]         # boussinesq, exp_coeff, general, poro, brinkman
    out_par = [print_dump, 0, 0, 0, 0, 0, 0,
               1, 1, 1, 0, 0, 0, 0, 0, 0, 0]

    print(f"[pyIB2d] {name}: Nx={grid[0]} Ny={grid[1]} dt={time[1]:g} "
          f"Tfinal={Tfinal:g} steps={int(round(Tfinal / time[1]))} "
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
