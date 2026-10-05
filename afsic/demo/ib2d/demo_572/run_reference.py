"""Run the IB2d reference solution of "Channel Flow Cellular Race" with pyIB2d.

    python run_reference.py [--tend T] [--dt DT] [--print-dump N]
                            [--out FILE] [--ib2d DIR] [--run DIR]

Pure Python, no MATLAB.  The example is one of the ``matIB2d`` ones (1-based
identifiers); the identifier columns are rebased into the scratch run
directory together with the ported ``please_Compute_External_Forcing.py`` (the
same module ``main.py`` imports, so both codes use one forcing implementation).
The massive-boundary (ghost) history, which pyIB2d never writes to VTK, is
recorded by wrapping ``please_Update_Massive_Boundary_Velocity``; a replay of
the ported ghost ODE driven by pyIB2d's own markers is printed as a check of
the port.

The VTK output plus the ghost history are converted to ``ib2d_reference.npz``
by ``ib2d_reference.py``.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time as _time

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
    verbatim.  NOTE: a ``.mass`` row is ``id  k  M``, so it has **one** id
    column -- the ``id_cols`` table below is the fixed one (IB2d's own
    ``read_Mass_Points`` only ever interprets column 0 as an index).
    """
    with open(path_in) as fh:
        n = int(float(fh.readline().split()[0]))
        rows = np.atleast_2d(np.loadtxt(fh, max_rows=n))
    rows[:, :n_id_cols] -= index_base
    with open(path_out, "w") as fh:
        fh.write(f"{rows.shape[0]}\n")
        for r in rows:
            fh.write(" ".join(f"{v:.16e}" for v in r) + "\n")


def n_steps(Tfinal, dt):
    """Number of steps of the driver's ``while current_time < T_FINAL`` loop."""
    n, t = 0, 0.0
    while t < Tfinal:
        t += dt
        n += 1
    return n


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
    name = P.get("string_name", "rubberband")
    Tfinal = args.tend if args.tend is not None else float(P["Tfinal"])

    if not os.path.isdir(args.ib2d):
        subprocess.run(["git", "clone", "https://github.com/nickabattista/ib2d",
                        os.path.join(REPO, "third_party", "ib2d")], check=True)

    run = args.run
    if os.path.exists(run):
        shutil.rmtree(run)
    os.makedirs(run)
    # structure/field data files: 1-based identifiers are rebased, everything
    # else (vertex, tracers, concentration, porous, brinkman, ...) is copied
    # verbatim
    id_cols = {"spring": 2, "beam": 3, "target": 1, "muscle": 2, "mass": 1}
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
        if fn.endswith(".py"):
            shutil.copy(os.path.join(src, fn), run)
    shutil.copy(os.path.join(src, "input2d"), run)

    blackbox = os.path.join(args.ib2d, "IBM_Blackbox")
    assert os.path.exists(os.path.join(blackbox, "IBM_Driver.py")), \
        f"pyIB2d not found at {args.ib2d}"
    for p in (run, blackbox):
        if p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, blackbox)
    sys.path.insert(0, run)
    import IBM_Driver as Driver

    fluid = [float(P["mu"]), float(P["rho"])]
    nx = args.nx if args.nx is not None else int(P["Nx"])
    ny = int(round(nx * float(P["Ly"]) / float(P["Lx"])))
    grid = [nx, ny, float(P["Lx"]), float(P["Ly"]), int(P["supp"])]
    dt = args.dt if args.dt is not None else float(P["dt"])
    print_dump = (args.print_dump if args.print_dump is not None
                  else int(P["print_dump"]))
    time = [Tfinal, dt]
    # the driver's flag vector, taken from the example's own input2d (index
    # order: springs, update_springs, targets, update_target, beams,
    # update_beams, nonInv, update_nonInv, FV_LT muscle, 3-element muscle,
    # arb_ext_force, tracers, mass, gravity, xG, yG, porous, concentration,
    # electro, damped, update_damped, boussinesq, exp_coeff, general_force,
    # poroelastic, brinkman)
    def flag(name, default=0):
        return int(float(P.get(name, default)))

    mass_yes = flag("mass_pts")
    n_mass_file = 0
    mass_file = os.path.join(run, f"{name}.mass")
    if mass_yes and os.path.exists(mass_file):
        with open(mass_file) as fh:
            n_mass_file = int(float(fh.readline().split()[0]))

    if mass_yes and n_mass_file == 1:
        # UPSTREAM BUG (pyIB2d IBM_Driver.py, "only 1 mass point" branch):
        #   mass_info[0,1] = xLag[mass_info[0].astype('int')]
        # indexes xLag with the whole 5-column row (id, 0, 0, 0, 0) instead of
        # the id mass_info[0,0], i.e. it assigns a 5-vector into a scalar and
        # raises ValueError.  Make the reader hand the (correct) general branch
        # two rows by appending a k = 0 dummy ghost (on a *different* marker:
        # ``give_Me_Mass_Lagrangian_Force_Densities`` writes the mass force with
        # ``fx[IDs] = fx[IDs] + ...``, so a duplicated id would let the dummy's
        # zero overwrite the real spring force).  With zero stiffness the dummy
        # produces no force anywhere; only its own free-fall state, which is
        # discarded below.
        _real_read_mass = Driver.read_Mass_Points

        def _read_mass_2rows(struct_name, _real=_real_read_mass):
            m = np.atleast_2d(_real(struct_name)).astype(float)
            with open(struct_name + ".vertex") as fh:
                n_vertex = int(float(fh.readline().split()[0]))
            dummy = m[0:1].copy()
            dummy[0, 0] = (m[0, 0] + 1) % n_vertex
            dummy[0, 1] = 0.0
            return np.vstack([m, dummy])

        Driver.read_Mass_Points = _read_mass_2rows
        print("[pyIB2d] patched read_Mass_Points: 1-point .mass -> +k=0 dummy "
              "row (upstream single-mass-point branch is broken)")

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

    print(f"[pyIB2d] {name}: Nx={grid[0]} Ny={grid[1]} dt={time[1]:g} "
          f"Tfinal={Tfinal:g} steps={n_steps(Tfinal, dt)} "
          f"print_dump={out_par[0]} mass_pts={mass_yes}")

    # ------------------------------------------------------------------
    # ghost (massive boundary point) history: pyIB2d writes the markers to
    # VTK but never the ghosts, so wrap the routine that closes the mass block
    # of every step and record mass_info / mVelocity there.
    # ------------------------------------------------------------------
    mass_hist = {"step": [], "X": [], "V": []}
    _real_upd = Driver.please_Update_Massive_Boundary_Velocity
    _cnt = [0]
    # the per-step half-step marker positions the ghost ODE is driven by: used
    # for a replay check of the ported ghost algorithm (dumped after the run)
    _real_force = Driver.please_Find_Lagrangian_Forces_On_Eulerian_grid
    _replay = {"Xh": []}

    def _force(*a, **kw):
        out = _real_force(*a, **kw)
        if mass_yes:
            _replay["Xh"].append((np.array(a[2], dtype=float).copy(),
                                  np.array(a[3], dtype=float).copy()))
        return out

    def _upd(dt_step, mass_info, mVelocity, F_Mass_Bnd, gravity_Info):
        out = _real_upd(dt_step, mass_info, mVelocity, F_Mass_Bnd, gravity_Info)
        if abs(dt_step - dt) <= 1e-12 * max(1.0, dt):   # full-step call closes it
            _cnt[0] += 1
            mass_hist["step"].append(_cnt[0])
            mass_hist["X"].append(np.array(mass_info[:, 1:3], dtype=float))
            mass_hist["V"].append(np.array(out, dtype=float))
        return out

    if mass_yes:
        Driver.please_Update_Massive_Boundary_Velocity = _upd
        Driver.please_Find_Lagrangian_Forces_On_Eulerian_grid = _force

    cwd = os.getcwd()
    tic = _time.perf_counter()
    try:
        os.chdir(run)
        Driver.main(fluid, grid, time, lag, out_par, name)
    finally:
        os.chdir(cwd)
    print(f"[pyIB2d] driver wall time {_time.perf_counter() - tic:.1f} s")

    extra = None
    if mass_yes and mass_hist["step"]:
        ex = ib2d_io.load_example(src, index_base=1)
        ids0 = ex["mass"]["ids"]
        k0, M0 = ex["mass"]["k"], ex["mass"]["M"]
        n_real = ids0.size
        n_done = mass_hist["step"][-1]

        # ---- replay check ------------------------------------------------
        # Re-integrate the ported ghost algorithm (main.py's mass block) with
        # the very marker positions pyIB2d used, from the same initial state.
        # A round-off match proves the ported ghost ODE (sign, dt/2 ordering,
        # gravity) is exactly pyIB2d's -- so any difference left in the demo is
        # the fluid coupling, not the ghost algorithm.
        if len(_replay["Xh"]) >= n_done and bool(lag[13]):
            gx, gy = float(P.get("x_gravity_vec_comp", 0.0)), \
                float(P.get("y_gravity_vec_comp", -1.0))
            gn = np.hypot(gx, gy)
            gvec = np.array([gx / gn, gy / gn]) if gn else np.zeros(2)
            mp = ex["X"][ids0].copy()
            mv = np.zeros((n_real, 2))
            nrm = 0.0
            for s in range(n_done):
                xh, yh = _replay["Xh"][s]
                mp_old = mp.copy()
                mh = mp + 0.5 * dt * mv                       # half-step move
                F = k0[:, None] * (mh - np.column_stack([xh[ids0], yh[ids0]]))
                acc = F / M0[:, None] - 9.80665 * gvec[None, :]
                mp = mp_old + dt * (mv - 0.5 * dt * acc)      # Move(dt, v_h)
                mv = mv - dt * acc                            # Update(dt, ...)
                nrm = max(nrm, np.abs(mp - mass_hist["X"][s][:n_real]).max())
            print(f"[pyIB2d] replay check of the ported ghost ODE driven by "
                  f"pyIB2d's own markers: max|dXm| = {nrm:.3e} m over {n_done} "
                  f"steps")
            del _replay["Xh"]

        # the frames the driver's VTK writer produced: the pre-loop initial
        # state plus, for every k >= 1 with k*print_dump + 1 <= n_done, the end
        # of that step
        frames = [0] + [k * print_dump + 1 for k in range(1, n_done + 1)
                        if k * print_dump + 1 <= n_done]
        Xg = np.zeros((len(frames), n_real, 2))
        Vg = np.zeros((len(frames), n_real, 2))
        Xg[0] = ex["X"][ids0]
        for j, s in enumerate(frames[1:], start=1):
            Xg[j] = mass_hist["X"][s - 1][:n_real]
            Vg[j] = mass_hist["V"][s - 1][:n_real]
        extra = dict(mass_ids=ids0, mass_step=np.array(frames, dtype=int),
                     Xm=Xg, Vm=Vg, n_mass=n_real,
                     mass_k=ex["mass"]["k"], mass_M=ex["mass"]["M"])
        print(f"[pyIB2d] recorded {len(frames)} ghost frames "
              f"({n_real} ghosts, {n_done} steps"
              + (", 1 dummy row dropped" if n_mass_file == 1 else "") + ")")

    viz = os.path.join(run, "viz_IB2d")
    assert os.path.isdir(viz), f"pyIB2d wrote no output ({viz} missing)"
    ib2d_reference.main(viz, args.out, dt=time[1], print_dump=out_par[0],
                        nx=grid[0], ny=grid[1], extra=extra)


if __name__ == "__main__":
    main()
