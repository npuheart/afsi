"""Quasi-static pressurized circular membrane for the AFSI solver (demo_442).

A thin elastic membrane (a closed 1-D curve) initialised in its circular
equilibrium configuration, centre (1/2, 1/2), radius R = 1/4, immersed in still
fluid in the unit square.  The membrane force density follows the paper
(Li et al. 2025, arXiv:2412.15408, and Gruninger & Griffith 2024):

    F(s, t) = kappa d^2 X / ds^2        (tension weak form: kappa |X_s|^2 / 2)

Because the initial state is in equilibrium, any deviation of the enclosed area
is a discretisation error of the coupling; the spurious vorticity (ideally zero)
probes the force-spreading quality of the IB4 kernel.

Difference from the paper: the fluid is solved with the AFSI finite-element
(Chorin projection) background and IB4 coupling only; the paper's periodic
domain is replaced by a closed box with no-slip walls (the membrane is far from
the walls and only short times are of interest).

Run (from this directory):

    conda activate afsi-dolfinx
    python main.py                      # N=128, MFAC=0.5, T=1 s (paper settings)
    N=64 T_END=0.5 python main.py       # quick
    MFAC=1.0 python main.py

Environment variables
---------------------
N        fluid cells per direction                                  [128]
MFAC     marker spacing / fluid grid spacing h                      [0.5]
T_END    final time [s]                                             [1.0]
KAPPA    membrane tension                                           [1.0]
R        membrane radius                                            [0.25]
OUT_EVERY  metric stride in steps                                   [8]
TAG      label for the output files                                 [default]
"""
import os

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np

import dolfinx
from dolfinx import log
from dolfinx.fem import (Function, functionspace, dirichletbc,
                         locate_dofs_topological, form, assemble_scalar,
                         Constant, Expression)
from dolfinx.mesh import CellType, GhostMode, locate_entities, create_mesh
from basix.ufl import element
from dolfinx.fem.petsc import create_vector, assemble_vector

from ufl import (Measure, TestFunction, SpatialCoordinate, as_vector,
                 dot, dx, inner, grad)

from afsic import ChorinSolver, IBMesh, IBInterpolation

comm = MPI.COMM_WORLD
rank = comm.rank

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N = int(os.environ.get("N", "128"))
MFAC = float(os.environ.get("MFAC", "0.5"))
T_END = float(os.environ.get("T_END", "1.0"))
KAPPA = float(os.environ.get("KAPPA", "1.0"))
R = float(os.environ.get("R", "0.25"))
OUT_EVERY = int(os.environ.get("OUT_EVERY", "8"))

LX = LY = 1.0
CX, CY = 0.5, 0.5
RHO_F = 1.0
MU_F = 1.0

h = LX / N
DT = h / 8.0                       # paper: dt = h/8
STEPS = int(round(T_END / DT))
M_MEM = max(8, int(round(2.0 * np.pi * R / (MFAC * h))))

tag = os.environ.get("TAG", f"N{N}_MFAC{MFAC:g}")
_demo_dir = os.path.dirname(os.path.abspath(__file__))
outdir = os.environ.get("OUTPUT_PATH", os.path.join(_demo_dir, "plot"))
os.makedirs(outdir, exist_ok=True)

if rank == 0:
    print(f"[demo_442] N={N} h={h:.5g} dt={DT:.5g} steps={STEPS} "
          f"markers={M_MEM} MFAC={MFAC} kappa={KAPPA} R={R}")

# ---------------------------------------------------------------------------
# Fluid: unit square, no-slip walls, pressure gauge at (0, 0)
# ---------------------------------------------------------------------------
mesh = dolfinx.mesh.create_rectangle(
    comm=comm, points=((0.0, 0.0), (LX, LY)), n=(N, N),
    cell_type=CellType.quadrilateral, ghost_mode=GhostMode.shared_facet)
mesh.topology.create_connectivity(1, 2)
fdim = mesh.topology.dim - 1

V = functionspace(mesh, element("Lagrange", mesh.topology.cell_name(), 2,
                                shape=(mesh.geometry.dim,)))
Q = functionspace(mesh, element("Lagrange", mesh.topology.cell_name(), 1))
V_io = functionspace(mesh, element("Lagrange", mesh.topology.cell_name(), 1,
                                   shape=(mesh.geometry.dim,)))

walls = locate_entities(mesh, fdim,
                        lambda x: np.logical_or(np.logical_or(
                            np.isclose(x[0], 0.0), np.isclose(x[0], LX)),
                            np.logical_or(np.isclose(x[1], 0.0),
                                          np.isclose(x[1], LY))))
u_zero = np.array((0.0,) * mesh.geometry.dim, dtype=PETSc.ScalarType)
bcu = [dirichletbc(u_zero, locate_dofs_topological(V, fdim, walls), V)]
point = dolfinx.mesh.locate_entities_boundary(
    mesh, 0, lambda x: np.logical_and(np.isclose(x[0], 0.0),
                                      np.isclose(x[1], 0.0)))
bcp = [dirichletbc(PETSc.ScalarType(0.0),
                   locate_dofs_topological(Q, 0, point), Q)]
ns_solver = ChorinSolver(V, Q, bcu, bcp, DT, RHO_F, MU_F)

# ---------------------------------------------------------------------------
# Membrane: closed 1-D mesh on the circle, tension weak form
# ---------------------------------------------------------------------------
theta = 2.0 * np.pi * np.arange(M_MEM) / M_MEM
circle_pts = np.column_stack([CX + R * np.cos(theta), CY + R * np.sin(theta)])
cells = np.column_stack([np.arange(M_MEM),
                         (np.arange(M_MEM) + 1) % M_MEM]).astype(np.int64)
coord_element = element("Lagrange", "interval", 1, shape=(2,))
membrane = create_mesh(comm, cells, coord_element, circle_pts)

Vs = functionspace(membrane, element("Lagrange", "interval", 2, shape=(2,)))
Vs_io = functionspace(membrane, element("Lagrange", "interval", 1, shape=(2,)))
assert Vs.dofmap.index_map_bs == 2

solid_coords = Function(Vs, name="solid_coords")
solid_coords_io = Function(Vs_io, name="solid_coords_io")
solid_force = Function(Vs, name="solid_force")
solid_velocity = Function(Vs, name="solid_velocity")
solid_coords.interpolate(lambda x: np.array([x[0], x[1]]))
ref = solid_coords.x.array.copy()

# regression guard: the FE line measure equals the chord polygon length
L_circle = assemble_scalar(
    form(Constant(membrane, 1.0) * Measure("dx", domain=membrane)))
L_poly = 2.0 * M_MEM * R * np.sin(np.pi / M_MEM)
assert abs(L_circle - L_poly) < 1e-12, \
    f"membrane length mismatch: {L_circle} vs {L_poly}"

dVs = TestFunction(Vs)
# psi(F) = kappa/2 |F|^2  ->  P = kappa F  ->  b1 = -kappa int X_s . dv_s ds
L_hat = form(-KAPPA * inner(grad(solid_coords), grad(dVs)) * dx)
b1 = create_vector(Vs)

# ---------------------------------------------------------------------------
# Immersed boundary coupling
# ---------------------------------------------------------------------------
ibmesh = IBMesh(0.0, LX, 0.0, LY, N, N, 2)
ib_interpolation = IBInterpolation(ibmesh)
coords_bg = Function(V)
coords_bg.interpolate(lambda x: np.array([x[0], x[1]]))
ibmesh.build_map(coords_bg._cpp_object)
ib_interpolation.evaluate_current_points(solid_coords._cpp_object)

# ---------------------------------------------------------------------------
# Metrics: enclosed area (from the P1 vertex coordinates, arc order), extreme
# fluid velocity, spurious vorticity, pressure jump across the membrane
# ---------------------------------------------------------------------------
dof_rows = Vs_io.tabulate_dof_coordinates()[:, :2]
vert_dof = []
for c in circle_pts:
    j = int(np.argmin(np.linalg.norm(dof_rows - c, axis=1)))
    assert np.linalg.norm(dof_rows[j] - c) < 1e-12
    vert_dof.append(j)
vert_dof = np.array(vert_dof)

def enclosed_area():
    xy = solid_coords_io.x.array.reshape(-1, 2)[vert_dof]
    x, y = xy[:, 0], xy[:, 1]
    xp, yp = np.roll(x, -1), np.roll(y, -1)
    return 0.5 * abs(np.sum(x * yp - xp * y))

solid_coords_io.interpolate(solid_coords)
A0 = enclosed_area()
A_poly = 0.5 * M_MEM * R ** 2 * np.sin(2.0 * np.pi / M_MEM)
assert abs(A0 - A_poly) < 1e-12, f"initial area {A0} vs polygon {A_poly}"

omega_expr = Expression(ns_solver.u_[1].dx(0) - ns_solver.u_[0].dx(1),
                        Q.element.interpolation_points)
omega = Function(Q, name="omega")

tree = dolfinx.geometry.bb_tree(mesh, mesh.geometry.dim)
def sample(func, x0):
    xp = np.array([x0[0], x0[1], 0.0], dtype=dolfinx.default_scalar_type)
    cells = dolfinx.geometry.compute_colliding_cells(
        mesh, dolfinx.geometry.compute_collisions_points(tree, xp), xp)
    if len(cells.array) == 0:
        return np.nan
    return func.eval(xp, cells.array[0])[0]

dx_fluid = Measure("dx", domain=mesh)
csv_path = os.path.join(outdir, f"metrics_{tag}.csv")
with open(csv_path, "w") as fh:
    fh.write("step,t,area,dA_rel,max_u,max_omega,p_in,p_out,p_jump\n")

log.set_log_level(log.LogLevel.WARNING)

def metrics(step, t):
    solid_coords_io.interpolate(solid_coords)
    A = enclosed_area()
    omega.interpolate(omega_expr)
    umax = float(np.max(np.abs(ns_solver.u_.x.array)))
    wmax = float(np.max(np.abs(omega.x.array)))
    p_in = sample(ns_solver.p_, (CX + 0.10, CY))
    p_out = sample(ns_solver.p_, (CX + 0.45, CY))
    row = (step, t, A, abs(A - A0) / A0, umax, wmax, p_in, p_out, p_in - p_out)
    with open(csv_path, "a") as fh:
        fh.write(",".join((str(row[0]),) + tuple(f"{v:.10g}" for v in row[1:]))
                 + "\n")
    if rank == 0:
        print(f"  step {step:6d} t={t:7.4f} area={A:.10f} dA={row[3]:.3e} "
              f"|u|={umax:.3e} |w|={wmax:.3e} dp={row[8]: .4f}")
    return row

# ---------------------------------------------------------------------------
# Time loop
# ---------------------------------------------------------------------------
for step in range(STEPS + 1):
    t = step * DT

    if step > 0:
        ns_solver.solve_one_step()

        ib_interpolation.fluid_to_solid(ns_solver.u_._cpp_object,
                                        solid_velocity._cpp_object)
        solid_coords.x.array[:] += solid_velocity.x.array[:] * DT
        solid_coords.x.scatter_forward()

        ib_interpolation.evaluate_current_points(solid_coords._cpp_object)
        with b1.localForm() as loc:
            loc.set(0)
        assemble_vector(b1, L_hat)
        b1.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        with b1.getBuffer() as arr:
            solid_force.x.array[: len(arr)] = arr[:]

        ib_interpolation.solid_to_fluid(ns_solver.f._cpp_object,
                                        solid_force._cpp_object)
        ns_solver.f.x.scatter_forward()

    if step % OUT_EVERY == 0 or step == STEPS:
        metrics(step, t)

# ---------------------------------------------------------------------------
# Final snapshot
# ---------------------------------------------------------------------------
solid_coords_io.interpolate(solid_coords)
omega.interpolate(omega_expr)
np.savez_compressed(
    os.path.join(outdir, f"snapshot_{tag}.npz"),
    fluid_x=V.tabulate_dof_coordinates()[:, :2],
    fluid_u=ns_solver.u_.x.array.reshape(-1, mesh.geometry.dim),
    p_x=Q.tabulate_dof_coordinates()[:, :2], p=ns_solver.p_.x.array,
    omega_x=Q.tabulate_dof_coordinates()[:, :2], omega=omega.x.array,
    membrane_x=Vs.tabulate_dof_coordinates()[:, :2],
    membrane_coords=solid_coords.x.array.reshape(-1, 2),
    membrane_ref=ref.reshape(-1, 2), circle=circle_pts,
    meta=np.array([N, M_MEM, MFAC, DT, T_END, KAPPA, R, A0]),
)
if rank == 0:
    print(f"[demo_442] done: {STEPS} steps, N={N} MFAC={MFAC}; csv={csv_path}")
