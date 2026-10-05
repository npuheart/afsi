"""IB2d coagulation examples in AFSI.

Ports of the ``matIB2d`` ``Example_Coagulation`` tree (see ``readme.md`` for the
exact upstream path of this demo) into AFSI:

* the fluid is the periodic Taylor-Hood/Peskin-RK2 solver of the other ib2d
  demos (``PeskinRK2Solver``), markers are carried by a zero-stiffness FE
  chain and coupled with ``afsic``'s ``IBMesh``/``IBInterpolation``;
* the Lagrangian structure is springs + invariant beams + **target points**
  whose x-anchors are stirred by the example's own
  ``update_Target_Point_Positions.m`` (ported to
  ``ib2d_input/update_Target_Point_Positions.py``, the very same module pyIB2d
  imports from the run directory);
* the **coagulation** model (``coagulation = 1``) is the matrix
  ``matIB2d/IBM_Blackbox/give_Me_Coagulation_Force_Densities.m``: the cells are
  the first markers of the ``.vertex`` file (``<name>.coagulation`` gives the
  start index, the bond radius, the bond stiffness, the fracture threshold and
  the markers per cell); pairs of cell *centres* closer than the bond radius get
  a linear bond spring between their closest markers, and old bonds rupture when
  the squared bond force exceeds the fracture threshold.  The port lives in
  ``ib2d_input/coagulation_model.py`` (MATLAB-faithful; the pyIB2d module of the
  same name is dead, broken code -- see that file's docstring) and is evaluated
  at the half-step marker positions, adding its force densities to the springs'
  before the single spread, exactly like the MATLAB driver.

Run::

    conda activate afsi-dolfinx
    python run_reference.py             # pyIB2d + the same coagulation model
    python main.py                      # AFSI
    python compare.py

Output (``OUTPUT_PATH``, default ``./plot``): XDMF fields +
``afsi_result.npz`` (t, marker positions, target anchors, bond state,
coagulation-force magnitude, u/p/f on the IB2d grid).
"""
import os
import sys
import time

import numpy as np
from mpi4py import MPI

import basix.ufl
import dolfinx
from dolfinx import fem
from dolfinx.mesh import CellType, GhostMode

from afsic import IBMesh, IBInterpolation
from afsic.euler.PeskinRK2Solver import PeskinRK2Solver

import ib2d_io

_here = os.path.dirname(os.path.abspath(__file__))
IB2D_EXAMPLE = os.environ.get("IB2D_EXAMPLE", os.path.join(_here, "ib2d_input"))

sys.path.insert(0, IB2D_EXAMPLE)
from coagulation_model import Coagulation                       # noqa: E402
from update_Target_Point_Positions import \
    update_Target_Point_Positions                               # noqa: E402

comm = MPI.COMM_WORLD
assert comm.size == 1, "this demo is serial (IB coupling of afsic gathers on rank 0)"

# the matIB2d example files number the Lagrangian points from 1
ex = ib2d_io.load_example(IB2D_EXAMPLE, index_base=1)
P = ex["params"]
rho, mu = float(P["rho"]), float(P["mu"])
Lx, Ly = float(P["Lx"]), float(P["Ly"])
Nx = int(os.environ.get("NX", P["Nx"]))
Ny = int(os.environ.get("NY", round(Nx * Ly / Lx)))
dt = float(os.environ.get("DT", P["dt"]))
T = float(os.environ.get("TFINAL", P["Tfinal"]))
print_dump = int(os.environ.get("PRINT_DUMP", P["print_dump"]))
# the pyIB2d driver advances while (current_time < Tfinal) with
# current_time += dt, so rounding noise can add one last step; replicate it so
# that the dump schedules of the two codes end on the same step
_n, _t = 0, 0.0
while _t < T:
    _t += dt
    _n += 1
num_steps = _n
out_dir = os.environ.get("OUTPUT_PATH", os.path.join(_here, "plot"))
os.makedirs(out_dir, exist_ok=True)

GRAD_DIV = float(os.environ.get("GRAD_DIV", "0"))

X_ib2d = ex["X"]
Nb = X_ib2d.shape[0]
sp = ex["springs"]
bm = ex["beams"]
tg = ex["targets"]
# coagulation: on when input2d says so; COAGULATION=1/0 overrides the flag
# (1 forces the model on for the control configurations that ship
# coagulation = 0, 0 is the ablation used to isolate the bond forces) --
# documented extra checks, not the shipped upstream setups
coag_on = int(P.get("coagulation", 0))
if os.environ.get("COAGULATION") is not None:
    coag_on = int(os.environ["COAGULATION"])
coag_spec = ex.get("coagulation")
if coag_on and coag_spec is None:
    _f = os.path.join(IB2D_EXAMPLE, f"{ex['name']}.coagulation")
    if os.path.exists(_f):
        coag_spec = ib2d_io.read_coagulation(_f)
if not coag_on:
    coag_spec = None
ds_ib2d = Lx / (2 * Nx)   # IB2d sets ds = Lx/(2Nx)

assert bool(int(P["springs"])) and bool(int(P["beams"])) \
    and bool(int(P["target_pts"])) and bool(int(P["update_target"]))
assert not int(P.get("arb_ext_force", 0)) and not int(P.get("tracers", 0)) \
    and not int(P.get("mass_pts", 0)) and not int(P.get("poroelastic", 0)) \
    and not int(P.get("concentration", 0)) \
    and not int(P.get("damped_springs", 0)) \
    and not int(P.get("nonInvariant_beams", 0)) \
    and not int(P.get("FV_LT_muscle", 0)) \
    and not int(P.get("user_force_model", 0)), \
    "this template implements springs, invariant beams, (stirred) target "\
    "points and the coagulation model only"

# target-point table, IB2d layout [id, x, y, k] (the .m moves column 1)
tg_info = np.empty((len(tg["ids"]), 4))
tg_info[:, 0] = tg["ids"]
tg_info[:, 1] = X_ib2d[tg["ids"], 0]
tg_info[:, 2] = X_ib2d[tg["ids"], 1]
tg_info[:, 3] = tg["k"]

coag = (None if coag_spec is None
        else Coagulation(coag_spec, Lx, Ly, index_base=1))

if comm.rank == 0:
    print(f"IB2d example : {IB2D_EXAMPLE}")
    print(f"rho={rho} mu={mu} box={Lx}x{Ly} grid={Nx}x{Ny} dt={dt} T={T} "
          f"steps={num_steps} ds={ds_ib2d} print_dump={print_dump}")
    print(f"structure    : {Nb} markers = {len(sp['conn'])} springs "
          f"(k={sp['k'].min():.3g}..{sp['k'].max():.3g}), "
          f"{len(bm['p1'])} invariant beams (kb={bm['kb'][0]:.3g}), "
          f"{len(tg['ids'])} stirred targets (k={tg['k'][0]:.3g})")
    if coag is None:
        print("coagulation  : off (no coagulation flag / file)")
    else:
        print(f"coagulation  : {len(coag.counts)} cells of "
              f"{coag.counts.min()}..{coag.counts.max()} markers, "
              f"threshold={coag.threshold:.6g} bond_k={coag.bond_k:g} "
              f"fracture={coag.fracture:g}")
    print(f"grad-div     : gamma={GRAD_DIV:g}")

# ---------------------------------------------------------------------------
# Fluid: periodic Taylor-Hood, Peskin two-stage scheme
# ---------------------------------------------------------------------------
mesh = dolfinx.mesh.create_rectangle(
    comm, ((0.0, 0.0), (Lx, Ly)), (Nx, Ny),
    cell_type=CellType.quadrilateral, ghost_mode=GhostMode.shared_facet)
solver = PeskinRK2Solver(mesh, (0.0, Lx, 0.0, Ly), dt, rho, mu, grad_div=GRAD_DIV)
Vc = solver.V

ibmesh = IBMesh(0.0, Lx, 0.0, Ly, Nx, Ny, 1)   # order 1: vertices = IB2d grid
V_ib = fem.functionspace(mesh, basix.ufl.element("Lagrange", "quadrilateral", 1,
                                                 shape=(2,)))
u_ib, f_ib = fem.Function(V_ib), fem.Function(V_ib)
coords_bg = fem.Function(V_ib)
coords_bg.interpolate(lambda x: np.array([x[0], x[1]]))
ibmesh.build_map(coords_bg._cpp_object)
ib = IBInterpolation(ibmesh)


def ib_velocity(u):
    """Eulerian velocity on the IB (vertex) grid."""
    u_ib.interpolate(u)
    return u_ib


# ---------------------------------------------------------------------------
# Structure: zero-stiffness FE chain carrying the marker dofs
# ---------------------------------------------------------------------------
cells = np.column_stack([np.arange(Nb - 1), np.arange(Nb - 1) + 1]).astype(np.int64)
coord_el = basix.ufl.element("Lagrange", "interval", 1, shape=(2,))
structure = dolfinx.mesh.create_mesh(comm, cells, coord_el, X_ib2d)
Vs = fem.functionspace(structure, coord_el)
assert Vs.dofmap.index_map.size_local == Nb

X = fem.Function(Vs, name="position")
X.interpolate(lambda x: np.array([x[0], x[1]]))
X_h = fem.Function(Vs, name="position_half")
X_h.x.array[:] = X.x.array
U_s = fem.Function(Vs, name="velocity")
F_s = fem.Function(Vs, name="force")

dof_rows = Vs.tabulate_dof_coordinates()[:, :2]
dof_of_ib2d = np.array([int(np.argmin(np.linalg.norm(dof_rows - p, axis=1)))
                        for p in X_ib2d])
assert len(np.unique(dof_of_ib2d)) == Nb
assert np.allclose(dof_rows[dof_of_ib2d], X_ib2d, atol=1e-12)


def marker_array(fn):
    return fn.x.array.reshape(-1, 2)[dof_of_ib2d]


def wrap_periodic(fn, dofs):
    """IB2d wraps marker coordinates into [0, L) after each update."""
    a = fn.x.array.reshape(-1, 2)
    a[dofs, 0] %= Lx
    a[dofs, 1] %= Ly


sp_a, sp_b = sp["conn"][:, 0], sp["conn"][:, 1]
sp_k, sp_L = sp["k"], sp["L"]
bm_p1, bm_p2, bm_p3 = bm["p1"], bm["p2"], bm["p3"]
bm_k, bm_C = bm["kb"], bm["C"]


def _min_image(d, L):
    """IB2d's minimum-image branch for a leader-follower difference."""
    over = np.abs(d) > L / 2
    d[over] = np.sign(d[over]) * (L - np.sign(d[over]) * d[over])
    return d


def _beam_boundary(Xp, Xq, Xr, L):
    """IB2d ``check_If_Beam_Points_Pass_Through_Boundary`` along one axis."""
    dX_pq, dX_qr = Xp - Xq, Xq - Xr
    over = np.abs(dX_pq) > 5 * ds_ib2d
    Xp = np.where(over, Xp + np.where(dX_pq < 0, L, -L), Xp)
    over = np.abs(dX_qr) > 5 * ds_ib2d
    Xr = np.where(over, Xr + np.where(dX_qr < 0, -L, L), Xr)
    return Xp, Xq, Xr


def spring_forces(Xm, Fm):
    """Linear springs, F = k (L - L0) dhat (IB2d give_Me_Spring_...)."""
    d = Xm[sp_b] - Xm[sp_a]
    d[:, 0] = _min_image(d[:, 0], Lx)
    d[:, 1] = _min_image(d[:, 1], Ly)
    L = np.hypot(d[:, 0], d[:, 1])
    sF = sp_k * (L - sp_L) / L
    np.add.at(Fm[:, 0], sp_a, sF * d[:, 0])
    np.add.at(Fm[:, 1], sp_a, sF * d[:, 1])
    np.add.at(Fm[:, 0], sp_b, -sF * d[:, 0])
    np.add.at(Fm[:, 1], sp_b, -sF * d[:, 1])


def beam_forces(Xm, Fm):
    """Invariant (cross-product) beams, as in demo_446/455/478."""
    Xp = Xm[bm_p1].copy()
    Xq = Xm[bm_p2].copy()
    Xr = Xm[bm_p3].copy()
    Xp, Xq, Xr = _beam_boundary(Xp, Xq, Xr, Lx)
    for c in (0, 1):
        Xp[:, c], Xq[:, c], Xr[:, c] = _beam_boundary(Xp[:, c], Xq[:, c],
                                                      Xr[:, c], Ly)
    S = ((Xr[:, 0] - Xq[:, 0]) * (Xq[:, 1] - Xp[:, 1])
         - (Xr[:, 1] - Xq[:, 1]) * (Xq[:, 0] - Xp[:, 0]))
    K = bm_k * (S - bm_C)
    np.add.at(Fm[:, 0], bm_p1, K * (Xr[:, 1] - Xq[:, 1]))
    np.add.at(Fm[:, 1], bm_p1, -K * (Xr[:, 0] - Xq[:, 0]))
    np.add.at(Fm[:, 0], bm_p2, K * ((Xq[:, 1] - Xp[:, 1]) + (Xr[:, 1] - Xq[:, 1])))
    np.add.at(Fm[:, 1], bm_p2, -K * ((Xr[:, 0] - Xq[:, 0]) + (Xq[:, 0] - Xp[:, 0])))
    np.add.at(Fm[:, 0], bm_p3, K * (Xq[:, 1] - Xp[:, 1]))
    np.add.at(Fm[:, 1], bm_p3, -K * (Xq[:, 0] - Xp[:, 0]))


def target_forces(Xm, Fm):
    """IB2d target-point force densities ``k (X_anchor - X)`` with the IB2d
    minimum-image fold; the anchors are moved by the example's .m each step."""
    d = tg_info[:, 1:3] - Xm[tg["ids"]]
    d[:, 0] = _min_image(d[:, 0], Lx)
    d[:, 1] = _min_image(d[:, 1], Ly)
    np.add.at(Fm, tg["ids"], tg_info[:, 3:4] * d)


def coag_forces(Xm, Fm):
    """Coagulation bond force densities (IB2d give_Me_Coagulation_...)."""
    fxc, fyc = coag.forces(Xm)
    Fm[:, 0] += fxc
    Fm[:, 1] += fyc
    return np.abs(np.hypot(fxc, fyc)).max() if fxc.size else 0.0


# ---------------------------------------------------------------------------
# Output sampling on the IB2d grid
# ---------------------------------------------------------------------------
Vdof = Vc.tabulate_dof_coordinates()[:, :2]
iu = np.rint(Vdof[:, 0] / (Lx / Nx)).astype(int)
ju = np.rint(Vdof[:, 1] / (Ly / Ny)).astype(int)
on_grid = lambda Q, i, j: (np.isclose(Q[:, 0], i * (Lx / Nx))    # noqa: E731
                           & np.isclose(Q[:, 1], j * (Ly / Ny)))
keep = (iu < Nx) & (ju < Ny) & on_grid(Vdof, iu, ju)
Q2s = fem.functionspace(mesh, ("Lagrange", 2))
p_grid_fn = fem.Function(Q2s)
Pdof = Q2s.tabulate_dof_coordinates()[:, :2]
ip = np.rint(Pdof[:, 0] / (Lx / Nx)).astype(int)
jp = np.rint(Pdof[:, 1] / (Ly / Ny)).astype(int)
keep_p = (ip < Nx) & (jp < Ny) & on_grid(Pdof, ip, jp)

V1 = fem.functionspace(mesh, basix.ufl.element("Lagrange", "quadrilateral", 1,
                                               shape=(2,)))
u_io, f_io = fem.Function(V1, name="u"), fem.Function(V1, name="f")
p_io = fem.Function(fem.functionspace(mesh, ("Lagrange", 1)), name="p")
# every vertex of the Q1 grid coincides with an IB2d node
Qcoords = V1.tabulate_dof_coordinates()[:, :2]
gi = np.rint(Qcoords[:, 0] / (Lx / Nx)).astype(int) % Nx
gj = np.rint(Qcoords[:, 1] / (Ly / Ny)).astype(int) % Ny


def grid_fields():
    u = np.zeros((Ny, Nx, 2))
    u[ju[keep], iu[keep]] = solver.u_.x.array.reshape(-1, 2)[keep]
    p_grid_fn.interpolate(solver.p_)
    p = np.zeros((Ny, Nx))
    p[jp[keep_p], ip[keep_p]] = p_grid_fn.x.array[keep_p]
    f_io.interpolate(solver.f)
    f = np.zeros((Ny, Nx, 2))
    f[gj, gi] = f_io.x.array.reshape(-1, 2)
    return u, p, f


xf = dolfinx.io.XDMFFile(comm, os.path.join(out_dir, "fluid.xdmf"), "w")
xs = dolfinx.io.XDMFFile(comm, os.path.join(out_dir, "structure.xdmf"), "w")
xf.write_mesh(mesh)
xs.write_mesh(structure)
disp = fem.Function(Vs, name="displacement")
X0 = X.x.array.copy()


def write_output(t):
    u_io.interpolate(solver.u_)
    f_io.interpolate(solver.f)
    p_io.interpolate(solver.p_)
    for fn in (u_io, p_io, f_io):
        xf.write_function(fn, t)
    disp.x.array[:] = X.x.array - X0
    xs.write_function(disp, t)


def bonds_flat():
    return coag.snapshot() if coag is not None else np.zeros((0, 5))


hist = {"t": [0.0], "X": [marker_array(X).copy()],
        "Xa": [tg_info[:, 1].copy()], "fcoag": [0.0],
        "nbonds": [bonds_flat().shape[0]], "bonds": [bonds_flat()]}
u0, p0, f0 = grid_fields()
hist["u"], hist["p"], hist["f"] = [u0], [p0], [f0]
write_output(0.0)
if comm.rank == 0:
    print(f"initial      : markers in x [{X_ib2d[:, 0].min():.3f}, "
          f"{X_ib2d[:, 0].max():.3f}], bonds={len(bonds_flat())}")

# ---------------------------------------------------------------------------
# Time loop -- Peskin (2002) / IB2d IBM_Driver ordering
# ---------------------------------------------------------------------------
tic = time.perf_counter()
ctime = 0.0
for step in range(num_steps):
    # (1) half-step Lagrangian positions X_h = X + dt/2 U^n(X)
    ib.evaluate_current_points(X._cpp_object)
    if step == 0:
        X_h.x.array[:] = X.x.array
    else:
        ib.fluid_to_solid(ib_velocity(solver.u_n)._cpp_object, U_s._cpp_object)
        X_h.x.array[:] = X.x.array + 0.5 * dt * U_s.x.array

    # (2b) stir: advance the target anchors (the driver's
    #      ``update_Target_Point_Positions`` sees ``current_time`` accumulated
    #      by repeated ``+= dt``, exactly as here)
    tg_info = update_Target_Point_Positions(dt, ctime, tg_info)

    # (3) forces at X_h, spread with the 4-point kernel
    Fm = np.zeros((Nb, 2))
    Xm = marker_array(X_h)
    spring_forces(Xm, Fm)
    beam_forces(Xm, Fm)
    target_forces(Xm, Fm)
    fcoag = coag_forces(Xm, Fm) if coag is not None else 0.0
    Fdof = np.empty_like(Fm)
    Fdof[dof_of_ib2d] = Fm
    F_s.x.array[:] = Fdof.ravel()
    F_s.x.array[:] *= ds_ib2d
    ib.evaluate_current_points(X_h._cpp_object)
    ib.solid_to_fluid(f_ib._cpp_object, F_s._cpp_object)
    solver.periodic_fold(f_ib)
    if f_ib is not solver.f:
        solver.f.interpolate(f_ib)

    # (4) two-stage fluid solve: u_h, u^{n+1}, p^{n+1/2}
    solver.solve_one_step()

    # (5) markers
    ib.fluid_to_solid(ib_velocity(solver.u_h)._cpp_object, U_s._cpp_object)
    X.x.array[:] += dt * U_s.x.array
    X.x.scatter_forward()
    wrap_periodic(X, dof_of_ib2d)

    ctime += dt
    # Dump schedule: the pyIB2d driver dumps when its 0-based step counter
    # ``cter`` equals a positive multiple of print_dump, i.e. after
    # print_dump, 2*print_dump, ... completed steps (plus the initial state
    # before the loop).  Mirror exactly that so frame k of both codes is the
    # same step.
    if (step + 1) % print_dump == 0 or step + 1 == num_steps:
        t = (step + 1) * dt
        hist["t"].append(t)
        hist["X"].append(marker_array(X).copy())
        hist["Xa"].append(tg_info[:, 1].copy())
        hist["fcoag"].append(fcoag)
        hist["nbonds"].append(bonds_flat().shape[0])
        hist["bonds"].append(bonds_flat())
        ug, pg, fg = grid_fields()
        hist["u"].append(ug)
        hist["p"].append(pg)
        hist["f"].append(fg)
        write_output(t)
        if comm.rank == 0:
            print(f"step {step+1:6d}  t={t:.5f}  bonds={len(bonds_flat()):4d}  "
                  f"|f_coag|max={fcoag:.4e}  "
                  f"|Xa-{tg_info[0, 1]:.4f}|  "
                  f"umax={np.abs(ug).max():.3f}  "
                  f"wall={time.perf_counter()-tic:.1f}s", flush=True)

elapsed = time.perf_counter() - tic
xf.close()
xs.close()
allbonds = np.vstack(hist["bonds"])
fname = os.path.join(out_dir, "afsi_result.npz")
np.savez_compressed(fname,
                    t=np.array(hist["t"]), X=np.array(hist["X"]),
                    Xa=np.array(hist["Xa"]),
                    u=np.array(hist["u"]), p=np.array(hist["p"]),
                    f=np.array(hist["f"]),
                    nbonds=np.array(hist["nbonds"]), bonds=allbonds,
                    fcoag=np.array(hist["fcoag"]),
                    dx=Lx / Nx, dy=Ly / Ny, dt=dt, nx=Nx, ny=Ny)
if comm.rank == 0:
    print(f"done in {elapsed:.1f}s -> {fname}")
    if coag is not None:
        print(f"bonds formed {coag.n_formed}, fractured {coag.n_fractured}, "
              f"active {coag.aggregate.shape[0]}")
