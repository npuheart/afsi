"""Run the IB2d reference solution of the example in ``ib2d_input/``.

    python run_reference.py [--tend T] [--dt DT] [--print-dump N] [--nx NX]
                            [--src DIR] [--out FILE] [--run DIR]

Pure Python, no MATLAB.  The example files are copied verbatim from
``ib2d_input/`` into a scratch run directory; the Lagrangian identifiers are
rebased to 0 there when the example uses MATLAB's 1-based numbering (detected
from the files themselves), and every ``.py`` module of ``ib2d_input/`` is
copied as well because pyIB2d imports its example-specific modules
(``please_Compute_External_Forcing``, ``update_Springs``, ...) from the run
directory.  The same ported modules are evaluated by ``main.py``, so both codes
share one implementation of every user-supplied model.

The VTK output of pyIB2d (fluid + structure) is converted to
``ib2d_reference.npz`` by ``ib2d_reference.py``.

pyIB2d quirks patched here (all documented in the demo readme):

* the tracer move call drops the four trailing arguments of the 15-argument
  ``please_Move_Lagrangian_Point_Positions`` signature;
* ``read_In_Brinkman_Info`` returns the undefined name ``Brink`` (NameError);
* the brinkman branch of the driver ``chdir``s twice into ``viz_IB2d``;
* ``update_nonInv_Beams`` is called with ``beams_info`` instead of the
  non-invariant array (the wrapper below feeds it the right one).
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


def shift_ids(path_in, path_out, index_base, n_id_cols):
    """Copy an IB2d structure file with its identifiers rebased to 0.

    The IB2d readers take the first ``n_id_cols`` columns as Lagrangian
    indices; every other column is copied verbatim.
    """
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
    name = P.get("string_name", "structure")
    Tfinal = args.tend if args.tend is not None else float(P["Tfinal"])

    if not os.path.isdir(args.ib2d):
        subprocess.run(["git", "clone", "https://github.com/nickabattista/ib2d",
                        os.path.join(REPO, "third_party", "ib2d")], check=True)

    index_base = ib2d_io.detect_index_base(src)
    run = args.run
    if os.path.exists(run):
        shutil.rmtree(run)
    os.makedirs(run)
    # structure/field data files: identifier columns are rebased when the
    # example is a matIB2d (1-based) one, everything else is copied verbatim
    for fn in sorted(os.listdir(src)):
        if not fn.startswith(name + "."):
            continue
        ext = fn.split(".", 1)[1]
        ncols = ib2d_io._ID_COLS.get(ext)
        if index_base and ncols:
            shift_ids(os.path.join(src, fn), os.path.join(run, fn),
                      index_base, ncols)
        else:
            shutil.copy(os.path.join(src, fn), run)
    # arbitrary auxiliary data files some user modules read (All_Positions.txt,
    # *.phases, ...) and the user modules themselves
    for fn in sorted(os.listdir(src)):
        if fn.endswith(".py") and fn != "run_reference.py":
            shutil.copy(os.path.join(src, fn), run)
        elif fn.endswith((".txt", ".dat", ".csv")) and not fn.startswith(name + "."):
            shutil.copy(os.path.join(src, fn), run)
    shutil.copy(os.path.join(src, "input2d"), run)

    blackbox = os.path.join(args.ib2d, "IBM_Blackbox")
    assert os.path.exists(os.path.join(blackbox, "IBM_Driver.py")), \
        f"pyIB2d not found at {args.ib2d}"
    sys.path.insert(0, blackbox)
    sys.path.insert(0, run)
    import IBM_Driver as Driver

    # pyIB2d bug: read_In_Brinkman_Info loads the matrix into ``C`` but returns
    # the undefined name ``Brink`` -> NameError for every brinkman example
    def _read_brink(struct_name):
        with open(os.path.join(os.getcwd(), struct_name + ".brinkman")) as fh:
            k = float(fh.readline().strip())
            return np.loadtxt(fh), k

    Driver.read_In_Brinkman_Info = _read_brink

    # pyIB2d bug: for brinkman runs the driver chdirs into viz_IB2d and then
    # into viz_IB2d *again* to dump the permeability map; pre-create it
    os.makedirs(os.path.join(run, "viz_IB2d", "viz_IB2d"), exist_ok=True)

    # pyIB2d port bug: its own tracer update call
    #     please_Move_Lagrangian_Point_Positions(Uh, Vh, xT, yT, xT, yT, x, y,
    #                                            dt, grid_Info, 0)
    # drops the four trailing arguments (porous_Yes..F_Poro) of the 15-argument
    # signature.  Patch a dispatcher (marker calls pass all 15 and fall through)
    # that re-inserts the tracer arguments; mu is only used on the poroelastic
    # branch.
    _real_move = Driver.please_Move_Lagrangian_Point_Positions
    # the concentration source needs the marker positions *at the scalar
    # update*, i.e. the X^{n+1} that the driver's step-4 marker move has just
    # produced (same state the AFSI port uses), so remember them here
    _last_X = {"x": None, "y": None}

    def _move_dispatch(*a):
        if len(a) == 11:
            Uh, Vh, xT, yT, xT2, yT2, x, y, dt, grid_Info, porous = a
            out = _real_move(0.0, Uh, Vh, xT, yT, xT2, yT2, x, y, dt,
                             grid_Info, porous, 0, None, None)
        else:
            out = _real_move(*a)
        if out is not None:
            _last_X["x"], _last_X["y"] = np.array(out[0]), np.array(out[1])
        return out

    Driver.please_Move_Lagrangian_Point_Positions = _move_dispatch

    def flag(nm, default=0):
        return int(float(P.get(nm, default)))

    # pyIB2d bug: the driver calls update_nonInv_Beams(dt, current_time,
    # beams_info) -- the *invariant*-beam array (a placeholder when the example
    # has no invariant beams).  Wrap the ported module so that it always
    # receives the non-invariant array it expects.
    if flag("update_nonInv_beams") and flag("nonInvariant_beams"):
        import update_nonInv_Beams as _unb
        _orig_unb = _unb.update_nonInv_Beams
        with open(os.path.join(run, name + ".nonInv_beam")) as fh:
            fh.readline()
            _state = {"info": np.loadtxt(fh, usecols=(0, 1, 2, 3, 4, 5))}

        def _unb_patched(dt, t, _ignored, _orig=_orig_unb, _st=_state):
            _st["info"] = _orig(dt, t, _st["info"])
            return _st["info"]

        _unb.update_nonInv_Beams = _unb_patched

    fluid = [float(P["mu"]), float(P["rho"])]
    nx = args.nx if args.nx is not None else int(P["Nx"])
    ny = int(round(nx * float(P["Ly"]) / float(P["Lx"])))
    grid = [nx, ny, float(P["Lx"]), float(P["Ly"]), int(P["supp"])]
    dt = args.dt if args.dt is not None else float(P["dt"])
    _n, _t = 0, 0.0
    while _t < Tfinal:
        _t += dt
        _n += 1
    print_dump = (args.print_dump if args.print_dump is not None
                  else int(P.get("print_dump") or max(1, _n // 20)))
    time = [Tfinal, dt]
    # the driver's flag vector, taken from the example's own input2d (index
    # order: springs, update_springs, targets, update_target, beams,
    # update_beams, nonInv, update_nonInv, FV_LT muscle, 3-element muscle,
    # arb_ext_force, tracers, mass, gravity, xG, yG, porous, concentration,
    # electro, damped, update_damped, boussinesq, exp_coeff, general_force,
    # poroelastic, brinkman)
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
    # out_par: print_dump, plot_* (matplotlib, off), then the VTK saves that
    # ib2d_reference.py reads back: pressure, velocity vector, force
    out_par = [print_dump, 0, 0, 0, 0, 0, 0,
               1, 1, 1, 0, 0, 0, 0, 1, 1, 0]

    # ------------------------------------------------------------------
    # initial scalar: the shipped ``*.concentration`` files of this family
    # have **no** ``kDiff`` header line (unlike e.g. the bifurcating-artery
    # one), so pyIB2d's ``read_In_Concentration_Info`` -- and every MATLAB
    # reader -- either crashes or mis-reads the field.  Write the run copy
    # with the ``kDiff`` of ``input2d`` in front, exactly like main.py does,
    # and block-average it when ``--nx`` coarsens the grid.
    # ------------------------------------------------------------------
    kd = None
    if flag("concentration"):
        cfile = os.path.join(run, f"{name}.concentration")
        if os.path.exists(cfile):
            with open(cfile) as fh:
                first = fh.readline()
            try:
                kd, skip = float(first), 1
            except ValueError:
                kd, skip = float(P.get("kDiff", 0.0)), 0
                print(f"[pyIB2d] {name}.concentration has no kDiff header "
                      f"(upstream quirk); using kDiff={kd:g} from input2d")
            C0 = np.loadtxt(cfile, skiprows=skip)
            if C0.shape != (ny, nx):
                by, bx = C0.shape[0] // ny, C0.shape[1] // nx
                assert C0.shape[0] == ny * by and C0.shape[1] == nx * bx, \
                    f"initial scalar grid {C0.shape} not a multiple of {(ny, nx)}"
                print(f"[pyIB2d] block-averaged {name}.concentration "
                      f"{C0.shape} -> {(ny, nx)} ({by}x{bx} blocks)")
                C0 = C0.reshape(ny, by, nx, bx).mean(axis=(1, 3))
            with open(cfile, "w") as fh:
                fh.write(f"{kd!r}\n")
                for row in C0:
                    fh.write(" ".join(f"{v:.16e}" for v in row) + "\n")

    # ------------------------------------------------------------------
    # concentration source term: pyIB2d has none (its driver never reads
    # ``source``/``k_source``/``c_inf``), so splice in the ported MATLAB
    # routines -- see ib2d_input/please_Find_Source_For_Concentration.py
    # ------------------------------------------------------------------
    source_model = flag("source")
    if source_model:
        assert flag("concentration"), "source > 0 needs concentration = 1"
        assert flag("advection") == 0, (
            "the WENO advection branch of the source stepper (Adv_flag = 1) is "
            "not implemented by pyIB2d nor by the AFSI port")
        import please_Find_Source_For_Concentration as SRC
        ksource = float(P.get("k_source", 0.0))
        c_inf = float(P.get("c_inf", 1.0))
        Lx_, Ly_ = float(P["Lx"]), float(P["Ly"])
        # the 9-entry MATLAB ``grid_Info`` (pyIB2d only receives the first 5
        # entries and expands them itself; the source routine needs all of them)
        with open(os.path.join(run, f"{name}.vertex")) as fh:
            try:
                nb_ = int(float(fh.readline().split()[0]))
            except ValueError:
                fh.seek(0)
                nb_ = len(np.atleast_2d(np.loadtxt(fh)))
        _g = [nx, ny, Lx_, Ly_, Lx_ / nx, Ly_ / ny, int(P["supp"]), nb_,
              Lx_ / (2 * nx)]

        def _upd_with_source(C, dt_, dx_, dy_, U, V, kDiff, _m=source_model,
                             _k=ksource, _ci=c_inf, _Lx=Lx_, _Ly=Ly_):
            f = SRC.please_Find_Source_For_Concentration(
                dt_, 0.0, _last_X["x"], _last_X["y"], None, None, _g, _m, _k,
                C, _ci, 0, 0)
            return SRC.please_Update_Adv_Diff_Concentration_Source(
                C, dt_, dx_, dy_, U, V, kDiff, f, _Lx, _Ly, Adv_flag=0)

        Driver.please_Update_Adv_Diff_Concentration = _upd_with_source
        print(f"[pyIB2d] concentration source spliced in: model={source_model} "
              f"k_source={ksource:g} c_inf={c_inf:g} (mode: un-split upwind)")

    print(f"[pyIB2d] {name}: Nx={grid[0]} Ny={grid[1]} dt={time[1]:g} "
          f"Tfinal={Tfinal:g} steps={_n} print_dump={out_par[0]} "
          f"index_base={index_base}")

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
