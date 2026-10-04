"""IB2d example "Moving Rubberband" (matIB2d
``Examples/Example_Moving_Rubberband``) in AFSI.

A 64-marker circular rubberband (semi-axes 0.2 x 0.2 around (0.25, 0.25),
``N = 2 Nx`` points) sits in a 1 x 1 periodic box of viscous fluid
(mu = 10, rho = 1, 32 x 32).  The band carries **no springs and no beams** --
every marker is a target point with k = 1e7 -- so it is a rigid *carrier*
whose anchors are prescribed in time by the example's
``update_Target_Point_Positions.m``:

    y(t) += (s/2) dt        on every step        (constant upward drift)
    x(t) += +s dt           while t mod 0.05 < 0.025   (right)
    x(t) -=  s dt           while t mod 0.05 >= 0.025  (left)

with ``period = 0.05`` s, ``L = 0.5`` m and ``s = L/(period/2) = 20`` m/s, so
over the example's ``Tfinal = 0.05`` s (dt = 1e-4, 500 steps) the band sweeps
once to the right and back while drifting 0.5 m upward: it ends centred at
(0.25, 0.75).  This is the standard IB2d teaching case for **dynamically
updated target points** -- the structure drags the fluid through stiff
tethers, and since the anchors move 1e-3 m per step (6.4% of the marker
spacing ds = 1/64) the port also exercises a fast-moving immersed boundary.

The example is one of the ``matIB2d`` ones: its ``.target``/``.vertex`` files
are 1-based, so ``ib2d_io.load_example(..., index_base=1)`` rebases the
identifiers, and ``run_reference.py`` writes a rebased copy of the ``.target``
file for the pyIB2d reference driver.

AFSI ingredients (same as the other ib2d demos):

* fluid: periodic Q2/Q1 Taylor-Hood + Peskin (2002) two-stage scheme
  (``PeskinRK2Solver``);
* structure: a zero-stiffness FE chain carrying the marker dofs;
* forces: the IB2d target-point formula ``k (X_anchor - X^{n+1/2})`` evaluated
  at the half-step positions, with the anchors advanced by the *same* python
  port of the ``.m`` file that pyIB2d imports, at the same place in the step
  (after the half-step positions, before the forces).

Run::

    conda activate afsi-dolfinx
    python main.py                      # full example (T = 0.05 s, 32x32)
    TFINAL=0.002 python main.py         # smoke test

Output (``OUTPUT_PATH``, default ``./plot``): XDMF fields +
``afsi_result.npz`` (t, X markers, X_anchor, u/p on the IB2d grid) with the
same layout as the pyIB2d reference from ``run_reference.py``.
"""
import os
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
num_steps = int(round(T / dt))
out_dir = os.environ.get("OUTPUT_PATH", os.path.join(_here, "plot"))
os.makedirs(out_dir, exist_ok=True)

GRAD_DIV = float(os.environ.get("GRAD_DIV", "0"))
FIELD_EVERY = int(os.environ.get("FIELD_EVERY", "50"))   # u/p sampling (steps)

# the moving-target law (port of update_Target_Point_Positions.m)
PERIOD = 0.05                     # time it takes to move to the right
TRAVEL = 0.5                      # distance the band moves in each direction
SPEED = TRAVEL / (PERIOD / 2)

X_ib2d = ex["X"]
Nb = X_ib2d.shape[0]
tg = ex["targets"]
ds_ib2d = min(Lx / (2 * Nx), Ly / (2 * Ny))

assert "springs" not in ex and "beams" not in ex and "noninv_beams" not in ex \
    and "mass" not in ex and "muscles" not in ex and "d_springs" not in ex, \
    "this demo implements the moving-target example only"
assert tg["ids"].size == Nb and np.array_equal(tg["ids"], np.arange(Nb)), \
    "the moving-target law of this example advances every marker in place"

if comm.rank == 0:
    print(f"IB2d example : {IB2D_EXAMPLE}")
    print(f"rho={rho} mu={mu} box={Lx}x{Ly} grid={Nx}x{Ny} dt={dt} T={T} "
          f"steps={num_steps} ds={ds_ib2d}")
    print(f"structure    : {Nb} markers, {len(tg['ids'])} target points "
          f"(k={tg['k'][0]:.2g}), speed s={SPEED:g} m/s")
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


tgt_ids, tgt_k = tg["ids"], tg["k"]
tgt_anchor = X_ib2d[tgt_ids].copy()      # moving: updated by update_targets()


def update_targets(t):
    """Port of the example's ``update_Target_Point_Positions.m``.

    Called once per step with the time at the *start* of the step, i.e. at the
    same place pyIB2d calls the ``.py`` twin of the ``.m`` file (after the
    half-step marker positions are formed and before the forces are spread).
    """
    tgt_anchor[:, 1] += (SPEED / 2) * dt
    step = SPEED * dt if np.remainder(t, PERIOD) < PERIOD / 2 else -SPEED * dt
    tgt_anchor[:, 0] += step


def target_forces(Xm, Fm):
    """IB2d target-point force densities ``k (X_anchor - X)``.

    IB2d folds the difference of a point that wrapped around the periodic box
    back into the box (``|d| > L/2`` branch of
    ``give_Me_Target_Lagrangian_Force_Densities``); reproduced here verbatim
    (it never fires in this example: the anchors track the markers to within
    a few 1e-6 m).
    """
    d = tgt_anchor - Xm[tgt_ids]
    for c, L in enumerate((Lx, Ly)):
        over = np.abs(d[:, c]) > L / 2
        sgn = np.sign(d[over, c])
        d[over, c] = sgn * (L - sgn * d[over, c])
    np.add.at(Fm, tgt_ids, tgt_k[:, None] * d)


def spread_forces_at_half():
    Fm = np.zeros((Nb, 2))
    target_forces(marker_array(X_h), Fm)
    Fdof = np.empty_like(Fm)
    Fdof[dof_of_ib2d] = Fm
    F_s.x.array[:] = Fdof.ravel()
    F_s.x.array[:] *= ds_ib2d
    ib.evaluate_current_points(X_h._cpp_object)
    ib.solid_to_fluid(f_ib._cpp_object, F_s._cpp_object)
    solver.periodic_fold(f_ib)
    if f_ib is not solver.f:
        solver.f.interpolate(f_ib)


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


def grid_fields():
    u = np.zeros((Ny, Nx, 2))
    u[ju[keep], iu[keep]] = solver.u_.x.array.reshape(-1, 2)[keep]
    p_grid_fn.interpolate(solver.p_)
    p = np.zeros((Ny, Nx))
    p[jp[keep_p], ip[keep_p]] = p_grid_fn.x.array[keep_p]
    return u, p


V1 = fem.functionspace(mesh, basix.ufl.element("Lagrange", "quadrilateral", 1,
                                               shape=(2,)))
u_io, f_io = fem.Function(V1, name="u"), fem.Function(V1, name="f")
p_io = fem.Function(fem.functionspace(mesh, ("Lagrange", 1)), name="p")
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


def area(Xm):
    """Enclosed area of the closed marker chain (shoelace)."""
    x, y = Xm[:, 0], Xm[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


hist = {"t": [0.0], "X": [marker_array(X).copy()],
        "Xa": [tgt_anchor.copy()]}
u0, p0 = grid_fields()
hist["u"], hist["p"] = [u0], [p0]
write_output(0.0)
if comm.rank == 0:
    print(f"initial      : area={area(X_ib2d):.6f} (pi a b = "
          f"{np.pi * 0.2 * 0.2:.6f}), max|X-X_anchor|="
          f"{np.abs(marker_array(X) - tgt_anchor).max():.2e}")

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

    # (2) move the target points (the driver does this right after X_h)
    update_targets(ctime)

    # (3) target forces at X_h, spread with the 4-point kernel
    spread_forces_at_half()

    # (4) two-stage fluid solve: u_h, u^{n+1}, p^{n+1/2}
    solver.solve_one_step()

    # (5) markers
    ib.fluid_to_solid(ib_velocity(solver.u_h)._cpp_object, U_s._cpp_object)
    X.x.array[:] += dt * U_s.x.array
    X.x.scatter_forward()

    ctime += dt
    # Dump schedule: exactly the steps on which the pyIB2d driver writes its
    # VTK frames.  Its loop dumps when its step counter ``cter`` (0-based,
    # checked before it is incremented) is a positive multiple of
    # ``print_dump``, i.e. at the *end* of step ``cter + 1``; mirroring that
    # here makes AFSI frame k and reference frame k the same step of the
    # simulation (the driver's own time stamp for frame k, ``k*print_dump*dt``,
    # is one step behind the state it wrote -- an upstream labelling quirk).
    # t below is the true time of the dumped state.
    dumped = (step % print_dump == 0 and step >= print_dump) \
        or (step + 1 == num_steps)
    if dumped:
        t = (step + 1) * dt
        hist["t"].append(t)
        hist["X"].append(marker_array(X).copy())
        hist["Xa"].append(tgt_anchor.copy())
        ug, pg = grid_fields()
        hist["u"].append(ug)
        hist["p"].append(pg)
        write_output(t)
        if comm.rank == 0:
            Xm = hist["X"][-1]
            print(f"step {step+1:5d}  t={t:.5f}  "
                  f"Xc=({Xm[:, 0].mean():.4f},{Xm[:, 1].mean():.4f})  "
                  f"area={area(Xm):.6f}  "
                  f"|X-Xa|max={np.abs(Xm - tgt_anchor).max():.2e}  "
                  f"umax={np.abs(ug).max():.3f}  "
                  f"wall={time.perf_counter()-tic:.1f}s", flush=True)

elapsed = time.perf_counter() - tic
xf.close()
xs.close()
fname = os.path.join(out_dir, "afsi_result.npz")
np.savez_compressed(fname,
                    t=np.array(hist["t"]), X=np.array(hist["X"]),
                    Xa=np.array(hist["Xa"]),
                    u=np.array(hist["u"]), p=np.array(hist["p"]),
                    dx=Lx / Nx, dy=Ly / Ny)
if comm.rank == 0:
    print(f"done in {elapsed:.1f}s -> {fname}")
