"""Run the IB2d reference solution of a coagulation example, with bonds.

    python run_reference.py [--tend T] [--out FILE] [--ib2d DIR] [--run DIR]

Pure Python, no MATLAB.  The example is a ``matIB2d`` one (1-based
identifiers), so the structure files are copied from ``ib2d_input/`` with their
identifier columns rebased to 0 (MATLAB/pyIB2d read *all* rows of a structure
file and ignore the declared count -- two of the coagulation examples ship a
``.beam`` file whose declared count is stale -- and ``ib2d_io`` does the same).
The example's own ``update_Target_Point_Positions.py`` and the ported
``coagulation_model.py`` are copied into the scratch run directory because
pyIB2d imports its run-time user modules from there.

**The coagulation model is not part of pyIB2d.**  ``IBM_Driver.py`` never
imports ``give_Me_Coagulation_Force_Densities.py``, and that module is broken
anyway (see ``ib2d_input/coagulation_model.py``).  This reference therefore
patches pyIB2d's ``please_Find_Lagrangian_Forces_On_Eulerian_grid`` with a
wrapper that adds the coagulation bond forces to the Eulerian force density at
the same place the MATLAB driver does (inside the force assembly, at the
half-step marker positions, before the single delta-function spread -- the
spread uses pyIB2d's own
``give_Me_Delta_Function_Approximations_For_Force_Calc``), and drives the bond
state with the *same* ``coagulation_model`` the AFSI port uses.  So this
reference validates the AFSI **coupling** (force placement, ``ds`` scaling,
half-step evaluation, time integration) but not the bond model itself -- for
that, ``coagulation_model.analytic_self_test`` is run by ``compare.py``.

The VTK output of pyIB2d (fluid + structure) is converted to
``ib2d_reference.npz`` by ``ib2d_reference.py``; the per-frame bond state and
target anchors are appended to that npz here.
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
    """Copy an IB2d structure file with its identifiers rebased to 0.

    The IB2d readers (``springs[:,0]``, ``springs[:,1]``) take the first
    ``n_id_cols`` columns as Lagrangian indices; every other column is copied
    verbatim.
    """
    with open(path_in) as fh:
        n = int(float(fh.readline().split()[0]))
        rows = np.atleast_2d(np.loadtxt(fh))
    rows[:, :n_id_cols] -= index_base
    with open(path_out, "w") as fh:
        fh.write(f"{rows.shape[0]}\n")
        for r in rows:
            fh.write(" ".join(f"{v:.16e}" for v in r) + "\n")
    assert n == rows.shape[0] or True          # declared count may be stale


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
    ap.add_argument("--coagulation", choices=("auto", "on", "off"),
                    default="auto",
                    help="override the input2d coagulation flag (on forces the "
                         "model on for the configurations that ship it off; "
                         "off is the ablation that isolates the bond forces)")
    args = ap.parse_args()

    src = args.src
    P = ib2d_io.read_input2d(os.path.join(src, "input2d"))
    name = P.get("string_name", "channel")
    Tfinal = args.tend if args.tend is not None else float(P["Tfinal"])
    dt = args.dt if args.dt is not None else float(P["dt"])
    print_dump = (args.print_dump if args.print_dump is not None
                  else int(P["print_dump"]))

    if not os.path.isdir(args.ib2d):
        subprocess.run(["git", "clone", "https://github.com/nickabattista/ib2d",
                        os.path.join(REPO, "third_party", "ib2d")], check=True)

    run = args.run
    if os.path.exists(run):
        shutil.rmtree(run)
    os.makedirs(run)
    # structure/field data files: 1-based identifiers are rebased, everything
    # else (vertex, tracers, concentration, .coagulation, ...) is copied
    # verbatim
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
    # user modules the driver imports from the run directory
    for fn in sorted(os.listdir(src)):
        if fn.endswith(".py") and fn != "run_reference.py":
            shutil.copy(os.path.join(src, fn), run)
    shutil.copy(os.path.join(src, "input2d"), run)

    blackbox = os.path.join(args.ib2d, "IBM_Blackbox")
    assert os.path.exists(os.path.join(blackbox, "IBM_Driver.py")), \
        f"pyIB2d not found at {args.ib2d}"
    sys.path.insert(0, blackbox)
    sys.path.insert(0, run)
    import IBM_Driver as Driver
    import please_Find_Lagrangian_Forces_On_Eulerian_grid as FLEG
    import coagulation_model

    # --- coagulation: patch the force assembly -----------------------------
    coag = None
    coag_on = (int(P.get("coagulation", 0)) if args.coagulation == "auto"
               else args.coagulation == "on")
    if coag_on:
        spec = ib2d_io.read_coagulation(os.path.join(src, f"{name}.coagulation"))
        coag = coagulation_model.Coagulation(spec, float(P["Lx"]),
                                             float(P["Ly"]), index_base=1)
    rec = {"bonds": [], "Xa": [], "fcoag": []}
    real_fleg = Driver.please_Find_Lagrangian_Forces_On_Eulerian_grid

    def patched(*a):
        Fx, Fy, F_Mass, F_Lag, F_Poro = real_fleg(*a)
        fmax = 0.0
        if coag is not None:
            xLag, yLag = np.asarray(a[2]), np.asarray(a[3])
            x, y, grid_Info = a[6], a[7], a[8]
            fx_c, fy_c = coag.forces(np.column_stack([xLag, yLag]))
            fmax = float(np.hypot(fx_c, fy_c).max())
            if fx_c.any() or fy_c.any():
                ds = grid_Info[8]
                dX, dY = FLEG.give_Me_Delta_Function_Approximations_For_Force_Calc(
                    x, y, grid_Info, xLag, yLag)
                Fx = Fx + ((fx_c * ds) * dY) @ dX
                Fy = Fy + ((fy_c * ds) * dY) @ dX
        rec["bonds"].append(coag.snapshot() if coag is not None
                            else np.zeros((0, 5)))
        rec["Xa"].append(np.array(a[11])[:, 1].copy())
        rec["fcoag"].append(fmax)
        k = len(rec["bonds"])
        _diag = int(os.environ.get("DIAG", 200))
        if _diag and k % _diag == 0:
            xl = np.asarray(a[2])
            msg = (f"[diag] step {k:6d}  finite={np.isfinite(xl).all()}  "
                   f"|X|max={np.abs(xl).max():.4e}")
            if _diag < 200:      # detailed probe
                spr = np.asarray(a[10])
                d = np.hypot(xl[spr[:, 1].astype(int)] - xl[spr[:, 0].astype(int)],
                             np.asarray(a[3])[spr[:, 1].astype(int)]
                             - np.asarray(a[3])[spr[:, 0].astype(int)])
                msg += (f"  min_spring={np.nanmin(d):.3e}  "
                        f"|F_Lag|max={np.nanmax(np.abs(F_Lag)):.3e}  "
                        f"|Fx_grid|max={np.nanmax(np.abs(Fx)):.3e}")
            if coag is not None:
                C = coag.centers(np.column_stack([xl, np.asarray(a[3])]))
                d = [float(np.hypot(C[i, 0] - C[j, 0], C[i, 1] - C[j, 1]))
                     for i in range(len(C)) for j in range(i + 1, len(C))]
                msg += (f"  bonds {coag.aggregate.shape[0]:4d}  min cell "
                        f"distance {min(d):.5f} (threshold "
                        f"{coag.threshold:.5f})")
            print(msg, flush=True)
        return Fx, Fy, F_Mass, F_Lag, F_Poro

    Driver.please_Find_Lagrangian_Forces_On_Eulerian_grid = patched

    fluid = [float(P["mu"]), float(P["rho"])]
    nx = args.nx if args.nx is not None else int(P["Nx"])
    ny = int(round(nx * float(P["Ly"]) / float(P["Lx"])))
    grid = [nx, ny, float(P["Lx"]), float(P["Ly"]), int(P["supp"])]

    def flag(nm, default=0):
        return int(float(P.get(nm, default)))

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
           flag("user_force_model"), 0, 0]
    assert len(lag) == 26
    # out_par: print_dump, plot_* (off), then vorticity/pressure/uVec and
    # fX/fY, which ib2d_reference.py reads back
    out_par = [print_dump, 0, 0, 0, 0, 0, 0,
               0, 1, 1, 0, 0, 0, 0, 1, 1, 0]

    steps = 0
    _t = 0.0
    while _t < Tfinal:
        _t += dt
        steps += 1
    print(f"[pyIB2d] {name}: Nx={grid[0]} Ny={grid[1]} dt={dt:g} "
          f"Tfinal={Tfinal:g} steps={steps} print_dump={out_par[0]} "
          f"coagulation={'on' if coag is not None else 'off'}")

    cwd = os.getcwd()
    try:
        os.chdir(run)
        Driver.main(fluid, grid, [Tfinal, dt], lag, out_par, name)
    finally:
        os.chdir(cwd)

    viz = os.path.join(run, "viz_IB2d")
    assert os.path.isdir(viz), f"pyIB2d wrote no output ({viz} missing)"
    ib2d_reference.main(viz, args.out, dt=dt, print_dump=out_par[0],
                        nx=grid[0], ny=grid[1])

    # --- append the bond / target-anchor history ---------------------------
    data = dict(np.load(args.out))
    nb = len(data["t"])
    frames = [(k * print_dump - 1) if k else None for k in range(nb)]
    assert len(rec["bonds"]) >= steps, \
        f"expected {steps} force calls, recorded {len(rec['bonds'])}"
    bonds = [np.zeros((0, 5)) if f is None else rec["bonds"][f]
             for f in frames]
    data["nbonds"] = np.array([b.shape[0] for b in bonds])
    data["bonds"] = np.vstack(bonds) if bonds else np.zeros((0, 5))
    data["fcoag"] = np.array([0.0 if f is None else rec["fcoag"][f]
                              for f in frames])
    data["Xa"] = np.array([rec["Xa"][0] if f is None else rec["Xa"][f]
                           for f in frames])
    np.savez_compressed(args.out, **data)
    print(f"bonds: frames {data['nbonds'].tolist()}, "
          f"formed {coag.n_formed if coag is not None else 0}, "
          f"fractured {coag.n_fractured if coag is not None else 0}")


if __name__ == "__main__":
    main()
