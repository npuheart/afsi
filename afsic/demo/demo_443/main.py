"""Compression test (rectangular block) for the AFSI solver (demo_443).

Quasi-static plane-strain compression of a 20 x 10 cm neo-Hookean block centred
in a 40 x 40 cm fluid box, after Wells et al. (2023), as used in the composite
B-spline paper (Li et al. 2025, arXiv:2412.15408).  Difference of this port: the
fluid is solved with the finite-element (Chorin projection) background of AFSI
and the IB coupling uses the four-point (IB4) kernel only.

Geometry (cm)

    y=25   top:    zero HORIZONTAL displacement (penalty); downward traction
                   200 dyn/cm over the central 10 cm (x in [15, 25])
    y=15   bottom: zero VERTICAL displacement (penalty)
    block x in [10, 30]; fluid box [0, 40] x [0, 40], all walls u = 0.
    Probe: vertical displacement of the top-centre material point (20, 25).

Run (from this directory):

    conda activate afsi-dolfinx
    python main.py                          # M=16, MFAC=0.5, ramp 40 s, to 100 s
    M=32 MFAC=0.5 python main.py
    M=16 MFAC=0.5 TL=4 TF=10 python main.py # quick

Environment variables
---------------------
M          solid elements along the 20 cm direction                [16]
MFAC       mesh factor: N_fluid = ceil(M * MFAC)                   [0.5]
N          override the fluid resolution directly                  [auto]
TL         load ramp time [s]                                      [40.0]
TF         final time [s]                                          [100.0]
DT_FACTOR  dt = DT_FACTOR * h                                      [0.001]
BETA_MULT  multiplier on the constraint penalty of the paper       [1.0]
OUT_EVERY  metric stride in steps                                  [50]
TAG        label for the output files                              [default]
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
from dolfinx.mesh import CellType, GhostMode, locate_entities, meshtags
from basix.ufl import element
from dolfinx.fem.petsc import create_vector, assemble_vector

from ufl import (Measure, TestFunction, SpatialCoordinate, as_vector,
                 dot, dx, inner, grad, det)

from afsic import ChorinSolver, IBMesh, IBInterpolation
from materials import NeoHookean

comm = MPI.COMM_WORLD
rank = comm.rank

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
M = int(os.environ.get("M", "16"))
MFAC = float(os.environ.get("MFAC", "0.5"))
TL = float(os.environ.get("TL", "40.0"))
TF = float(os.environ.get("TF", "100.0"))
DT_FACTOR = float(os.environ.get("DT_FACTOR", "0.001"))
BETA_MULT = float(os.environ.get("BETA_MULT", "1.0"))
MODEL = os.environ.get("MODEL", "standard")  # 'standard' (validated) or 'flory'
OUT_EVERY = int(os.environ.get("OUT_EVERY", "50"))

L = 40.0                        # fluid box side [cm]
X0b, X1b = 10.0, 30.0           # block extent in x [cm]
Y0b, Y1b = 15.0, 25.0           # block extent in y [cm]
LOAD_HALF = 5.0                 # load covers x in [20-5, 20+5] [cm]
T_LOAD = 200.0                  # traction density [dyn/cm]

RHO_F = 1.0                     # fluid density [g/cm^3]
MU_F = 0.16                     # fluid viscosity [dyn s / cm^2]
G = 80.194                      # solid shear modulus [dyn/cm^2]
KAPPA_MULT = float(os.environ.get("KAPPA_MULT", "1.0"))
# The reference enforces J = 1 through the discrete divergence-free coupling
# (its volumetric energy is only a stabiliser, "technically redundant").  The
# FE-coupled variant here leaks a little volume, so the material's volumetric
# stiffness is what actually pins J; KAPPA_MULT scales it to approach the
# incompressible limit of the reference.
KAPPA_STAB = KAPPA_MULT * 374.239   # numerical bulk modulus [dyn/cm^2]
PROBE = np.array([20.0, Y1b])

N = int(os.environ.get("N", "0"))
if N <= 0:
    # the paper uses N = ceil(M * MFAC); we take twice that so the IB4
    # kernel (4 cells wide) stays resolved over the 20 cm block
    N = int(np.ceil(2.0 * M * MFAC))
h = L / N
DT = DT_FACTOR * h
STEPS = int(round(TF / DT))
# paper: kappa_S = 2.5 * (2.5 dx / dt)
BETA = BETA_MULT * 2.5 * (2.5 * h / DT)
MR = max(1, M // 2)             # 20 x 10 block -> half the rows

tag = os.environ.get("TAG", f"M{M}_MFAC{MFAC:g}_N{N}")
_demo_dir = os.path.dirname(os.path.abspath(__file__))
outdir = os.environ.get("OUTPUT_PATH", os.path.join(_demo_dir, "plot"))
os.makedirs(outdir, exist_ok=True)

if rank == 0:
    print(f"[demo_443] M={M} MR={MR} MFAC={MFAC} N={N} h={h:.4g} "
          f"dt={DT:.4g} steps={STEPS} TL={TL} TF={TF} beta={BETA:.4g}")

# ---------------------------------------------------------------------------
# Fluid mesh and boundary conditions (u = 0 on all walls)
# ---------------------------------------------------------------------------
mesh = dolfinx.mesh.create_rectangle(
    comm=comm, points=((0.0, 0.0), (L, L)), n=(N, N),
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
                            np.isclose(x[0], 0.0), np.isclose(x[0], L)),
                            np.logical_or(np.isclose(x[1], 0.0),
                                          np.isclose(x[1], L))))
u_zero = np.array((0.0,) * mesh.geometry.dim, dtype=PETSc.ScalarType)
bcu = [dirichletbc(u_zero, locate_dofs_topological(V, fdim, walls), V)]
point = dolfinx.mesh.locate_entities_boundary(
    mesh, 0, lambda x: np.logical_and(np.isclose(x[0], 0.0),
                                      np.isclose(x[1], 0.0)))
bcp = [dirichletbc(PETSc.ScalarType(0.0),
                   locate_dofs_topological(Q, 0, point), Q)]
ns_solver = ChorinSolver(V, Q, bcu, bcp, DT, RHO_F, MU_F)

# ---------------------------------------------------------------------------
# Solid: structured rectangle mesh, directional penalty constraints, load
# ---------------------------------------------------------------------------
structure = dolfinx.mesh.create_rectangle(
    comm=comm, points=((X0b, Y0b), (X1b, Y1b)), n=(M, MR),
    cell_type=CellType.quadrilateral, ghost_mode=GhostMode.shared_facet)
structure.topology.create_connectivity(1, 2)

A_solid = assemble_scalar(form(Constant(structure, 1.0)
                               * Measure("dx", domain=structure)))
assert abs(A_solid - 200.0) < 1e-8 * 200.0, f"block area {A_solid} vs 200"

# facet tags: 1 = bottom (vertical constraint), 3 = top (horizontal
# constraint), 2 = loaded part of the top (central 10 cm).  The loaded facets
# are removed from the tag-3 set: a facet must appear only once in meshtags.
f_bottom = locate_entities(structure, 1, lambda x: np.isclose(x[1], Y0b))
f_top_all = locate_entities(structure, 1, lambda x: np.isclose(x[1], Y1b))
f_load = locate_entities(
    structure, 1,
    lambda x: np.logical_and(np.isclose(x[1], Y1b),
                             np.abs(x[0] - 20.0) <= LOAD_HALF + 1e-9))
f_top = np.setdiff1d(f_top_all, f_load)
marked = np.hstack([f_bottom, f_top, f_load])
values = np.hstack([np.full_like(f_bottom, 1), np.full_like(f_top, 3),
                    np.full_like(f_load, 2)])
order = np.argsort(marked)
sfacet_tag = meshtags(structure, 1, marked[order], values[order])
ds = Measure("ds", domain=structure, subdomain_data=sfacet_tag)

Vs = functionspace(structure, element("Lagrange",
                                      structure.topology.cell_name(), 2,
                                      shape=(structure.geometry.dim,)))
Vs_io = functionspace(structure, element("Lagrange",
                                         structure.topology.cell_name(), 1,
                                         shape=(structure.geometry.dim,)))
VJ = functionspace(structure, element("Lagrange",
                                      structure.topology.cell_name(), 1))
assert Vs.dofmap.index_map_bs == structure.geometry.dim

solid_coords = Function(Vs, name="solid_coords")
solid_coords_io = Function(Vs_io, name="solid_coords_io")
solid_force = Function(Vs, name="solid_force")
solid_velocity = Function(Vs, name="solid_velocity")
solid_coords.interpolate(lambda x: np.array([x[0], x[1]]))
ref = solid_coords.x.array.copy()

dVs = TestFunction(Vs)
X0 = SpatialCoordinate(structure)
material = NeoHookean(mu_s=G, lambda_s=KAPPA_STAB, model=MODEL)
PK1 = material.first_piola_kirchhoff_stress_v1(structure, solid_coords)

beta_c = Constant(structure, PETSc.ScalarType(BETA))
traction = Constant(structure, PETSc.ScalarType(0.0))

L_hat = -inner(PK1, grad(dVs)) * dx
# bottom: vertical spring to the reference position (beta * dY * dv_y)
L_hat -= beta_c * (solid_coords[1] - X0[1]) * dVs[1] * ds(1)
# top: horizontal spring to the reference position.  The paper prescribes
# zero horizontal displacement along the ENTIRE top boundary, including the
# loaded central 10 cm, so the spring acts on ds(3) and ds(2).
L_hat -= beta_c * (solid_coords[0] - X0[0]) * dVs[0] * ds(3)
L_hat -= beta_c * (solid_coords[0] - X0[0]) * dVs[0] * ds(2)
# applied load (spread force = F_ext + F_elastic, so the traction ADDS)
L_hat += inner(dVs, as_vector((0.0, -traction))) * ds(2)
L_hat = form(L_hat)
b1 = create_vector(Vs)

J_func = Function(VJ, name="J")
J_interp = Expression(det(grad(solid_coords)), VJ.element.interpolation_points)

# ---------------------------------------------------------------------------
# Immersed boundary coupling
# ---------------------------------------------------------------------------
ibmesh = IBMesh(0.0, L, 0.0, L, N, N, 2)
ib_interpolation = IBInterpolation(ibmesh)
coords_bg = Function(V)
coords_bg.interpolate(lambda x: np.array([x[0], x[1]]))
ibmesh.build_map(coords_bg._cpp_object)
ib_interpolation.evaluate_current_points(solid_coords._cpp_object)

# ---------------------------------------------------------------------------
# Probes and metrics
# ---------------------------------------------------------------------------
# solid_coords was initialised as the identity map, so ref.reshape(-1, bs)
# gives every node's REFERENCE coordinate in exactly the array order of
# solid_coords.x.array (2*nodes interleaved: entry of node n is [bs*n, bs*n+1]).
# (tabulate_dof_coordinates() rows are ordered differently - cell by cell -
# and must not be used to index into x.array.)
bs = structure.geometry.dim
ref_nodes = ref.reshape(-1, bs)
probe_node = int(np.argmin(np.linalg.norm(ref_nodes - PROBE, axis=1)))
probe_y = bs * probe_node + 1
assert np.linalg.norm(ref_nodes[probe_node] - PROBE) < 1e-6, \
    "probe node not found"

dx_fluid = Measure("dx", domain=mesh)
csv_path = os.path.join(outdir, f"metrics_{tag}.csv")
with open(csv_path, "w") as fh:
    fh.write("step,t,top_dY,max_disp,J_min,J_max,uL2,traction\n")

log.set_log_level(log.LogLevel.WARNING)

def metrics(step, t):
    disp = solid_coords.x.array.reshape(-1, bs) - ref.reshape(-1, bs)
    mag = np.linalg.norm(disp, axis=1)
    J_func.interpolate(J_interp)
    jv = J_func.x.array
    uL2 = assemble_scalar(form(dot(ns_solver.u_, ns_solver.u_) * dx_fluid)) ** 0.5
    row = (step, t, solid_coords.x.array[probe_y] - ref[probe_y],
           mag.max(), jv.min(), jv.max(), uL2, float(traction.value))
    with open(csv_path, "a") as fh:
        fh.write(",".join((str(row[0]),) + tuple(f"{v:.10g}" for v in row[1:]))
                 + "\n")
    if rank == 0:
        print(f"  step {step:7d} t={t:8.4f} dY={row[2]: .6f} "
              f"max|d|={row[3]:.4f} J=[{row[4]:.4f},{row[5]:.4f}]")
    return row

# ---------------------------------------------------------------------------
# Time loop
# ---------------------------------------------------------------------------
for step in range(STEPS + 1):
    t = step * DT
    traction.value = PETSc.ScalarType(T_LOAD * min(t / TL, 1.0))

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
J_func.interpolate(J_interp)
# array-order coordinates of the fluid velocity dofs (identity map), needed
# to pair column data with nodal values (tabulate rows are cell-ordered)
fluid_cx = Function(V)
fluid_cx.interpolate(lambda x: np.array([x[0], x[1]]))
# nodal vorticity field (spurious-velocity diagnostic)
V_om = functionspace(mesh, element("Lagrange",
                                   mesh.topology.cell_name(), 1))
omega = Function(V_om, name="omega")
om_interp = Expression(
    ns_solver.u_[1].dx(0) - ns_solver.u_[0].dx(1),
    V_om.element.interpolation_points)
omega.interpolate(om_interp)
# structured (i, j) index of every solid node on the (2M+1) x (2MR+1) P2 grid
step_x, step_y = (X1b - X0b) / (2 * M), (Y1b - Y0b) / (2 * MR)
solid_ij = np.column_stack([
    np.rint((ref_nodes[:, 0] - X0b) / step_x).astype(np.int64),
    np.rint((ref_nodes[:, 1] - Y0b) / step_y).astype(np.int64)])
np.savez_compressed(
    os.path.join(outdir, f"snapshot_{tag}.npz"),
    fluid_x=V.tabulate_dof_coordinates()[:, :2],
    fluid_u=ns_solver.u_.x.array.reshape(-1, mesh.geometry.dim),
    fluid_cx=fluid_cx.x.array.reshape(-1, 2),
    p_x=Q.tabulate_dof_coordinates()[:, :2], p=ns_solver.p_.x.array,
    omega=omega.x.array, omega_x=V_om.tabulate_dof_coordinates()[:, :2],
    solid_x=ref_nodes, solid_coords=solid_coords.x.array.reshape(-1, bs),
    solid_ref=ref_nodes, solid_ij=solid_ij,
    J=J_func.x.array, J_x=VJ.tabulate_dof_coordinates()[:, :2],
    meta=np.array([M, MR, MFAC, N, DT, TL, TF, T_LOAD, float(BETA)]),
)
if rank == 0:
    print(f"[demo_443] done: {STEPS} steps, M={M} MFAC={MFAC} N={N}; "
          f"csv={csv_path}")
