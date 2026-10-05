"""Run the IB2d reference solution of the ``Electromechanical_Pumping_w_Ca_Dynamics`` example.

    python run_reference.py [--tend T] [--dt DT] [--nx NX] [--print-dump N]
                            [--out FILE] [--ib2d DIR] [--run DIR]

Pure Python (no MATLAB).  The example is a ``matIB2d`` one (1-based
identifiers), so the structure files are rebased into the scratch run directory
and the example's own user modules (*.py ports, in ``ib2d_input/``) are copied
there -- pyIB2d imports them from the run directory.

**Electrophysiology.**  pyIB2d reads the ``electro_phys`` flag but never
implements the mechanism (see ``ib2d_input/update_Springs.py``): the MATLAB
driver overwrites the *stiffness* of springs ``ePhys_Start:ePhys_End`` with
``(16 v)^4`` every step.  To obtain a *fluid-structure* reference, this script
switches the documented ``update_springs`` hook on and lets
``ib2d_input/update_Springs.py`` inject exactly that modulation, so the
reference uses the same electrophysiology code (``FitzHugh_Nagumo_1d.py``) and
the same coupling as ``main.py``.  The flag vector is therefore the example's
own ``input2d`` with ``update_springs -> 1``.

``IB2D_DT`` / ``IB2D_TFINAL`` are exported for the ported user modules so that
``--dt`` / ``--tend`` also change the pre-computed potential consistently.
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


def shift_ids(path_in, path_out, index_base=1, n_id_cols=2):
    """Copy an IB2d structure file with its identifiers rebased to 0."""
    with open(path_in) as fh:
        n = int(float(fh.readline().split()[0]))
        rows = np.atleast_2d(np.loadtxt(fh, max_rows=n))
    rows[:, :n_id_cols] -= index_base
    with open(path_out, "w") as fh:
        fh.write(f"{rows.shape[0]}\n")
        for r in rows:
            fh.write(" ".join(f"{v:.16e}" for v in r) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tend", type=float, default=None, help="override Tfinal (s)")
    ap.add_argument("--dt", type=float, default=None, help="override dt (s)")
    ap.add_argument("--ephys-tend", type=float, default=None,
                    help="horizon of the *electrophysiology* model (defaults to "
                         "--tend: the calcium stimulus is periodic in physical "
                         "time, so shortening it only truncates both models)")
    ap.add_argument("--print-dump", type=int, default=None,
                    help="override print_dump (steps between dumps)")
    ap.add_argument("--nx", type=int, default=None,
                    help="override Nx (Ny follows the box aspect ratio)")
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
    name = P.get("string_name", "HeartTube")
    Tfinal = args.tend if args.tend is not None else float(P["Tfinal"])
    dt = args.dt if args.dt is not None else float(P["dt"])
    # the electrophysiology is pre-computed a priori by the example: its own
    # horizon is kept even when the fluid run is shortened (the .m's pulse
    # spacing is Tfinal/10)
    # the calcium stimulus period (0.5 s) is fixed in physical time, so the
    # electrophysiology horizon may follow the (possibly reduced) run horizon
    ephys_tend = (args.ephys_tend if args.ephys_tend is not None else Tfinal)

    if not os.path.isdir(args.ib2d):
        subprocess.run(["git", "clone", "https://github.com/nickabattista/ib2d",
                        os.path.join(REPO, "third_party", "ib2d")], check=True)

    run = args.run
    if os.path.exists(run):
        shutil.rmtree(run)
    os.makedirs(run)
    id_cols = {"spring": 2, "beam": 3, "target": 1, "muscle": 2, "mass": 2}
    for fn in sorted(os.listdir(src)):
        if not fn.startswith(name + "."):
            continue
        ext = fn.split(".", 1)[1]
        if ext in id_cols:
            shift_ids(os.path.join(src, fn), os.path.join(run, fn),
                      n_id_cols=id_cols[ext])
        else:
            shutil.copy(os.path.join(src, fn), run)
    for fn in sorted(os.listdir(src)):
        if fn.endswith(".py") and fn != "run_reference.py":
            shutil.copy(os.path.join(src, fn), run)
    shutil.copy(os.path.join(src, "input2d"), run)

    # the ported user modules build the electrophysiology from these
    os.environ["IB2D_DT"] = repr(dt)
    os.environ["IB2D_TFINAL"] = repr(ephys_tend)

    blackbox = os.path.join(args.ib2d, "IBM_Blackbox")
    assert os.path.exists(os.path.join(blackbox, "IBM_Driver.py")), \
        f"pyIB2d not found at {args.ib2d}"
    sys.path.insert(0, blackbox)
    sys.path.insert(0, run)
    import IBM_Driver as Driver

    fluid = [float(P["mu"]), float(P["rho"])]
    nx = args.nx if args.nx is not None else int(P["Nx"])
    ny = int(round(nx * float(P["Ly"]) / float(P["Lx"])))
    grid = [nx, ny, float(P["Lx"]), float(P["Ly"]), int(P["supp"])]
    print_dump = (args.print_dump if args.print_dump is not None
                  else int(P["print_dump"]))
    time = [Tfinal, dt]

    def flag(name_, default=0):
        return int(float(P.get(name_, default)))

    lag = [flag("springs"), flag("update_springs"), flag("target_pts"),
           flag("update_target"), flag("beams"), flag("update_beams"),
           flag("nonInvariant_beams"), flag("update_nonInv_beams"),
           flag("FV_LT_muscle"), flag("3_element_muscle"), flag("arb_ext_force"),
           flag("tracers"), flag("mass_pts"), flag("gravity"),
           float(P.get("x_gravity_vec_comp", 0.0)),
           float(P.get("y_gravity_vec_comp", -1.0)),
           flag("porous_media"), flag("concentration"), flag("electro_phys"),
           flag("damped_springs"), flag("update_damp_springs"),
           flag("boussinesq"), float(P.get("expansion_coeff", 1.0)),
           flag("user_force_model"), flag("poroelastic"), flag("brinkman")]
    assert len(lag) == 26
    # pyIB2d ignores electro_phys: drive the ported electro coupling through the
    # update_springs hook (same code path as the MATLAB driver's electro block)
    if lag[18] and not lag[8]:
        lag[1] = 1
    # out_par: print_dump, plot_* (off), save_Pressure, save_uVec
    out_par = [print_dump, 0, 0, 0, 0, 0, 0,
               0, 1, 1, 0, 0, 0, 0, 0, 0, 0]

    print(f"[pyIB2d] {name}: Nx={grid[0]} Ny={grid[1]} dt={time[1]:g} "
          f"Tfinal={Tfinal:g} steps={int(round(Tfinal / time[1]))} "
          f"print_dump={out_par[0]} electro={lag[18]} update_springs={lag[1]}")

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

    # record the applied electrophysiology (the stiffness the electro block
    # writes into the cross-tube springs) next to the fields: frame j of the
    # reference is the state at step j * print_dump, which is the row the
    # driver's ePhys_Ct counter reads at that step
    sys.path.insert(0, src)
    from FitzHugh_Nagumo_1d import FitzHugh_Nagumo_1d
    v_interp, ind_start, ind_end = FitzHugh_Nagumo_1d(time[1], ephys_tend)
    d = dict(np.load(args.out))
    d["k_elec"] = np.array([(16.0 * v_interp[j * out_par[0]]) ** 4
                            for j in range(len(d["t"]))])
    d["ephys_start"], d["ephys_end"] = ind_start, ind_end
    np.savez_compressed(args.out, **d)
    print(f"appended k_elec (springs {ind_start}..{ind_end}) to {args.out}")


if __name__ == "__main__":
    main()
