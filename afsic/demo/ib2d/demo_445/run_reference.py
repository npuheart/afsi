"""Run the IB2d reference solution with pyIB2d -- pure Python, no MATLAB.

    python run_reference.py [--tend T] [--out FILE] [--ib2d DIR] [--run DIR]

* reads the model parameters from ``ib2d_input/input2d`` (Nx, Ny, dt, Tfinal,
  print_dump, mu, rho; the structure comes from the ``jelly.*`` files),
* prepares ``ib2d_run/`` with those files plus a muscle driver
  ``update_Springs.py`` (the Python counterpart of the MATLAB example's
  ``update_Springs.m``; the muscle springs are auto-detected by F = 1e5),
* calls pyIB2d's ``IBM_Driver.main`` (the official Python port shipped in
  ``third_party/ib2d/pyIB2d``) with exactly the settings ``main.py`` uses,
* converts the VTK output to ``ib2d_reference.npz`` (layout: t, X, u, p, dx,
  dy -- the same as the other ib2d demos) via ``ib2d_reference.py``.

No MATLAB, Octave or the MATLAB sources are involved anywhere; the only
external piece is the pyIB2d Python code (cloned automatically by git if
``third_party/ib2d`` is missing).
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

UPDATE_SPRINGS_PY = '''"""Muscle drive of the Hoover jellyfish (auto-generated).

Counterpart of the MATLAB example's update_Springs.m: the muscle springs (the
rows with the contraction stiffness F = 1e5, written last by make_jelly.py)
get the resting length |cos(freq*pi*t)| -- two contractions per second
(freq = 2), i.e. one pulse every 0.5 s.
"""

import numpy as np

FREQ = 2.0
F_CONTRACTION = 1e5


def update_Springs(dt, current_time, xLag, yLag, springs_info):
    musc = np.abs(springs_info[:, 2] - F_CONTRACTION) < 1e-6
    springs_info[musc, 3] = np.abs(np.cos(FREQ * current_time * np.pi))
    return springs_info
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tend", type=float, default=None, help="override Tfinal (s)")
    ap.add_argument("--out", default=os.path.join(HERE, "ib2d_reference.npz"))
    ap.add_argument("--ib2d", default=os.path.join(REPO, "third_party", "ib2d", "pyIB2d"),
                    help="path of the pyIB2d directory (default third_party)")
    ap.add_argument("--run", default=os.path.join(HERE, "ib2d_run"))
    args = ap.parse_args()

    # --- model parameters (single source of truth: our generated input2d) --
    src = os.path.join(HERE, "ib2d_input")
    P = ib2d_io.read_input2d(os.path.join(src, "input2d"))
    Tfinal = args.tend if args.tend is not None else float(P["Tfinal"])

    if not os.path.isdir(args.ib2d):
        subprocess.run(["git", "clone", "https://github.com/nickabattista/ib2d",
                        os.path.join(REPO, "third_party", "ib2d")], check=True)

    # --- prepare the run directory -----------------------------------------
    run = args.run
    if os.path.exists(run):
        shutil.rmtree(run)
    os.makedirs(run)
    shutil.copy(os.path.join(src, "jelly.vertex"), run)
    # pyIB2d's own examples number the Lagrangian points from 0, while the
    # (MATLAB) IB2d examples -- and hence make_jelly.py -- use 1-based indices
    # in .spring / .target / .nonInv_beam.  Write 0-based copies for the
    # Python run (numerics unchanged, indices shifted by one).
    with open(os.path.join(src, "jelly.spring")) as fh:
        n = int(float(fh.readline().split()[0]))
        s = np.atleast_2d(np.loadtxt(fh, max_rows=n))
    s[:, 0] -= 1
    s[:, 1] -= 1
    with open(os.path.join(run, "jelly.spring"), "w") as fh:
        fh.write(f"{len(s)}\n")
        for r in s:
            fh.write(f"{int(r[0])} {int(r[1])} {r[2]:1.16e} {r[3]:1.16e} "
                     f"{int(r[4])}\n")
    with open(os.path.join(src, "jelly.target")) as fh:
        n = int(float(fh.readline().split()[0]))
        t = np.atleast_2d(np.loadtxt(fh, max_rows=n))
    t[:, 0] -= 1
    with open(os.path.join(run, "jelly.target"), "w") as fh:
        fh.write(f"{len(t)}\n")
        for r in t:
            fh.write(f"{int(r[0])} {r[1]:1.16e}\n")
    with open(os.path.join(src, "jelly.nonInv_beam")) as fh:
        n = int(float(fh.readline().split()[0]))
        b = np.atleast_2d(np.loadtxt(fh, max_rows=n))
    b[:, :3] -= 1
    with open(os.path.join(run, "jelly.nonInv_beam"), "w") as fh:
        fh.write(f"{len(b)}\n")
        for r in b:
            fh.write(f"{int(r[0])} {int(r[1])} {int(r[2])} {r[3]:1.16e} "
                     f"{r[4]:1.16e} {r[5]:1.16e}\n")
    with open(os.path.join(run, "update_Springs.py"), "w") as fh:
        fh.write(UPDATE_SPRINGS_PY)

    blackbox = os.path.join(args.ib2d, "IBM_Blackbox")
    assert os.path.exists(os.path.join(blackbox, "IBM_Driver.py")), \
        f"pyIB2d not found at {args.ib2d}"
    sys.path.insert(0, blackbox)
    sys.path.insert(0, run)
    import IBM_Driver as Driver

    # pyIB2d's IBM_Driver.main takes the same parameter lists as the MATLAB
    # input2d blocks (0-based):
    #   Fluid  [mu, rho]
    #   Grid   [Nx, Ny, Lx, Ly, supp]
    #   Time   [Tfinal, dt]
    #   LagStruct [springs, update_springs, target_pts, update_target,
    #              beams, update_beams, nonInv_beams, update_nonInv_beams,
    #              FV_LT_muscle, hill3, arb_ext, tracers, mass, gravity,
    #              xG, yG, porous, concentration, electro_phys,
    #              damped_springs, update_damped, boussinesq, exp_coeff,
    #              general_force, poroelastic, brinkman]
    #   Output [print_dump, plot_Matlab, plot_LagPts, plot_Velocity,
    #           plot_Vorticity, plot_MagVelocity, plot_Pressure,
    #           save_Vorticity, save_Pressure, save_uVec, save_uMag, save_uX,
    #           save_uY, save_fMag, save_fX, save_fY, save_hier]
    fluid = [float(P["mu"]), float(P["rho"])]
    grid = [int(P["Nx"]), int(P["Ny"]), float(P["Lx"]), float(P["Ly"]),
            int(P["supp"])]
    time = [Tfinal, float(P["dt"])]
    lag = [1, 1, 1, 0,               # springs, update_springs, targets, upd_target
           0, 0, 1, 0,               # beams, upd_beams, nonInv, upd_nonInv
           0, 0, 0, 0, 0, 0, 0.0, -1.0,   # muscle models, tracers, mass, gravity
           0, 0, 0, 0, 0,            # porous, concentration, electro, damped, upd
           0, 1.0, 0, 0, 0]          # boussinesq, exp_coeff, general, poro, brinkman
    out_par = [int(P["print_dump"]), 0, 0, 0, 0, 0, 0,
               1, 1, 1, 0, 0, 0, 0, 0, 0, 0]

    print(f"[pyIB2d] Nx={grid[0]} Ny={grid[1]} dt={time[1]:g} Tfinal={Tfinal:g} "
          f"steps={int(Tfinal / time[1])} print_dump={out_par[0]}")

    cwd = os.getcwd()
    try:
        os.chdir(run)
        Driver.main(fluid, grid, time, lag, out_par, P["string_name"])
    finally:
        os.chdir(cwd)

    # --- convert the VTK output to npz -------------------------------------
    viz = os.path.join(run, "viz_IB2d")
    assert os.path.isdir(viz), f"pyIB2d wrote no output ({viz} missing)"
    ib2d_reference.main(viz, args.out, dt=time[1], print_dump=out_par[0])


if __name__ == "__main__":
    main()
