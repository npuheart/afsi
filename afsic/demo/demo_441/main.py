"""Cook's membrane benchmark for the AFSI solver (demo_441).

Plane-strain benchmark after Wells et al. (2023), as used in the composite
B-spline paper (Li et al. 2025, arXiv:2412.15408) with a modified 13x13 cm
computational domain.  Difference of this port: the fluid is solved with the
finite-element (Chorin projection) background of AFSI and the IB coupling uses
the four-point (IB4) kernel only.

Geometry (cm), trapezoidal membrane immersed in the fluid, classic Cook layout:

    D (3.25, 7.9) -------- C (8.05, 9.5)      left edge  A-D : clamped (penalty)
      |                        |              right edge B-C : traction 6.25 dyn/cm
      |   solid Omega_0^s      |              bottom/top   : stress free
    A (3.25, 3.5) -------- B (8.05, 7.9)

Probe: vertical displacement of the material point that starts at the
upper-right corner C = (8.05, 9.5) cm.

Run (from this directory):

    conda activate afsi-dolfinx
    python main.py                          # M=8, MFAC=1, ramp 20 s, to 50 s (paper)
    M=16 MFAC=1 python main.py
    M=16 MFAC=1 TL=2 TF=4 python main.py    # quick, less quasi-static

Environment variables
---------------------
M          solid elements along the horizontal (4.8 cm) direction  [8]
MFAC       mesh factor: N_fluid = ceil(M * MFAC * 10/6.5)          [1.0]
N          override the fluid resolution directly                  [auto]
TL         load ramp time [s]                                      [20.0]
TF         final time [s]                                          [50.0]
DT_FACTOR  dt = DT_FACTOR * dx                                     [0.001]
BETA_MULT  multiplier on the clamp penalty of the paper            [1.0]
OUT_EVERY  CSV/metric stride in steps                              [50]
XDMF       1 = also write velocity/pressure/solid XDMF             [0]
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
from dolfinx.mesh import CellType, GhostMode, locate_entities, meshtags, create_mesh
from basix.ufl import element
from dolfinx.fem.petsc import create_vector, assemble_vector

from ufl import (Measure, TestFunction, SpatialCoordinate, FacetNormal,
                 as_vector, dot, dx, inner, grad, det)

from afsic import ChorinSolver, TimeManager, IBMesh, IBInterpolation
from materials import NeoHookean

comm = MPI.COMM_WORLD
rank = comm.rank

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
M = int(os.environ.get("M", "8"))
MFAC = float(os.environ.get("MFAC", "1.0"))
TL = float(os.environ.get("TL", "20.0"))
TF = float(os.environ.get("TF", "50.0"))
DT_FACTOR = float(os.environ.get("DT_FACTOR", "0.001"))
BETA_MULT = float(os.environ.get("BETA_MULT", "1.0"))
MODEL = os.environ.get("MODEL", "standard")  # 'standard' (validated) or 'flory'
OUT_EVERY = int(os.environ.get("OUT_EVERY", "50"))
WRITE_XDMF = bool(int(os.environ.get("XDMF", "0")))

# physical parameters (CGS, as in the paper)
Lx = Ly = 13.0                 # computational domain [cm]
RHO_F = 1.0                    # fluid density [g/cm^3]
MU_F = 0.16                    # fluid viscosity [dyn s / cm^2]
G = 83.333                     # solid shear modulus [dyn/cm^2]
KAPPA_MULT = float(os.environ.get("KAPPA_MULT", "1.0"))
# The reference enforces J = 1 through its discrete divergence-free coupling;
# the FE-coupled variant here leaks a little volume, so the material's own
# bulk modulus is what pins J (see demo_443).  The reported Cook's runs use
# the raw paper constant (KAPPA_MULT=1); this switch allows the calibration
# check.
KAPPA_STAB = KAPPA_MULT * 388.889   # numerical bulk modulus [dyn/cm^2]
T_LOAD = 6.25                  # traction density on the right edge [dyn/cm]

# geometry of the membrane [cm]
XA, YA = 3.25, 3.5             # A left-bottom
XB, YB = 8.05, 7.9             # B right-bottom
XC, YC = 8.05, 9.5             # C right-top (probe)
XD, YD = 3.25, 7.9             # D left-top
PROBE = np.array([XC, YC])

N = int(os.environ.get("N", "0"))
if N <= 0:
    N = int(np.ceil(M * MFAC * 10.0 / 6.5))
h = Lx / N                          # fluid grid spacing (NOT named dx: that
DT = DT_FACTOR * h                  # would shadow the UFL dx measure)
STEPS = int(round(TF / DT))
BETA = BETA_MULT * 0.125 * h / DT      # clamp penalty of the paper [dyn/cm^3]
MR = max(1, int(round(M * (YC - YA) / 6.5077)))   # ~equal element aspect ratio

tag = os.environ.get("TAG", f"M{M}_MFAC{MFAC:g}_N{N}")
_demo_dir = os.path.dirname(os.path.abspath(__file__))
outdir = os.environ.get("OUTPUT_PATH", os.path.join(_demo_dir, "plot"))
os.makedirs(outdir, exist_ok=True)

if rank == 0:
    print(f"[demo_441] M={M} MR={MR} MFAC={MFAC} N={N} h={h:.4g} "
          f"dt={DT:.4g} steps={STEPS} TL={TL} TF={TF} beta={BETA:.4g}")

# ---------------------------------------------------------------------------
# Fluid mesh, spaces and boundary conditions (u = 0 on all walls)
# ---------------------------------------------------------------------------
mesh = dolfinx.mesh.create_rectangle(
    comm=comm, points=((0.0, 0.0), (Lx, Ly)), n=(N, N),
    cell_type=CellType.quadrilateral, ghost_mode=GhostMode.shared_facet)
mesh.topology.create_connectivity(1, 2)
fdim = mesh.topology.dim - 1

v_cg2 = element("Lagrange", mesh.topology.cell_name(), 2,
                shape=(mesh.geometry.dim,))
s_cg1 = element("Lagrange", mesh.topology.cell_name(), 1)
V = functionspace(mesh, v_cg2)
Q = functionspace(mesh, s_cg1)
V_io = functionspace(mesh, element("Lagrange", mesh.topology.cell_name(), 1,
                                   shape=(mesh.geometry.dim,)))

walls = locate_entities(mesh, fdim,
                        lambda x: np.logical_or(np.logical_or(
                            np.isclose(x[0], 0.0), np.isclose(x[0], Lx)),
                            np.logical_or(np.isclose(x[1], 0.0),
                                          np.isclose(x[1], Ly))))
u_zero = np.array((0.0,) * mesh.geometry.dim, dtype=PETSc.ScalarType)
bcu = [dirichletbc(u_zero, locate_dofs_topological(V, fdim, walls), V)]

# pressure gauge: a single point value removes the null space
point = dolfinx.mesh.locate_entities_boundary(
    mesh, 0, lambda x: np.logical_and(np.isclose(x[0], 0.0),
                                      np.isclose(x[1], 0.0)))
bcp = [dirichletbc(PETSc.ScalarType(0.0),
                   locate_dofs_topological(Q, 0, point), Q)]

ns_solver = ChorinSolver(V, Q, bcu, bcp, DT, RHO_F, MU_F)

# ---------------------------------------------------------------------------
# Solid mesh: structured quads over the trapezoid, DOLFINx vertex ordering
# via the official VTK permutation (same guard as demo_423).
# ---------------------------------------------------------------------------
def trapezoid_mesh(mc, mr):
    pts = np.zeros(((mc + 1) * (mr + 1), 2), dtype=np.float64)
    for i in range(mc + 1):
        t = i / mc
        x = XA + t * (XB - XA)
        yb = YA + t * (YB - YA)
        yt = YD + t * (YC - YD)
        for j in range(mr + 1):
            eta = j / mr
            pts[i * (mr + 1) + j] = (x, yb + eta * (yt - yb))
    cells_vtk = np.zeros((mc * mr, 4), dtype=np.int64)
    c = 0
    for i in range(mc):
        for j in range(mr):
            v00 = i * (mr + 1) + j
            v10 = (i + 1) * (mr + 1) + j
            v11 = (i + 1) * (mr + 1) + j + 1
            v01 = i * (mr + 1) + j + 1
            cells_vtk[c] = [v00, v10, v11, v01]
            c += 1
    perm = np.asarray(dolfinx.cpp.io.perm_vtk(CellType.quadrilateral, 4),
                      dtype=np.int64)
    return pts, cells_vtk[:, perm]


pts, cells = trapezoid_mesh(M, MR)
coord_element = element("Lagrange", "quadrilateral", 1, shape=(2,))
structure = create_mesh(comm, cells, coord_element, pts)
structure.topology.create_connectivity(1, 2)

# regression guard: the FE area must equal 4.8 x (4.4 + 1.6)/2 = 14.4 cm^2
A_solid = assemble_scalar(form(Constant(structure, 1.0)
                               * Measure("dx", domain=structure)))
assert abs(A_solid - 14.4) < 1e-8 * 14.4 + 1e-10, \
    f"solid mesh area mismatch: {A_solid} vs 14.4"

# boundary facets: 1 = clamped left edge, 2 = loaded right edge
sfacets_left = locate_entities(structure, 1, lambda x: np.isclose(x[0], XA))
sfacets_right = locate_entities(structure, 1, lambda x: np.isclose(x[0], XB))
marked = np.hstack([sfacets_left, sfacets_right])
values = np.hstack([np.full_like(sfacets_left, 1),
                    np.full_like(sfacets_right, 2)])
order = np.argsort(marked)
sfacet_tag = meshtags(structure, 1, marked[order], values[order])
ds = Measure("ds", domain=structure, subdomain_data=sfacet_tag)

# spaces and fields on the solid
v_cg2_s = element("Lagrange", structure.topology.cell_name(), 2,
                  shape=(structure.geometry.dim,))
v_cg1_s = element("Lagrange", structure.topology.cell_name(), 1,
                  shape=(structure.geometry.dim,))
Vs = functionspace(structure, v_cg2_s)
Vs_io = functionspace(structure, v_cg1_s)
VJ = functionspace(structure, element("Lagrange",
                                      structure.topology.cell_name(), 1))
assert Vs.dofmap.index_map_bs == structure.geometry.dim

solid_coords = Function(Vs, name="solid_coords")
solid_coords_io = Function(Vs_io, name="solid_coords_io")
solid_force = Function(Vs, name="solid_force")
solid_force_io = Function(Vs_io, name="solid_force_io")
solid_velocity = Function(Vs, name="solid_velocity")

solid_coords.interpolate(lambda x: np.array([x[0], x[1]]))
ref = solid_coords.x.array.copy()          # reference coordinates

# ---------------------------------------------------------------------------
# Weak form: PK1 internal force + clamp penalty + ramped traction
# ---------------------------------------------------------------------------
dVs = TestFunction(Vs)
X0 = SpatialCoordinate(structure)
material = NeoHookean(mu_s=G, lambda_s=KAPPA_STAB, model=MODEL)
PK1 = material.first_piola_kirchhoff_stress_v1(structure, solid_coords)

beta_c = Constant(structure, PETSc.ScalarType(BETA))
traction = Constant(structure, PETSc.ScalarType(0.0))

L_hat = -inner(PK1, grad(dVs)) * dx
L_hat -= beta_c * inner(solid_coords - X0, dVs) * ds(1)
# spread force = F_ext + F_elastic: the external traction ADDS (+t) so the
# fluid is pulled in the load direction and the massless solid follows it
L_hat += inner(dVs, as_vector((0.0, traction))) * ds(2)
L_hat = form(L_hat)
b1 = create_vector(Vs)

# Jacobian field (deformed element volumes)
J_expr = det(grad(solid_coords))
J_func = Function(VJ, name="J")
J_interp = Expression(J_expr, VJ.element.interpolation_points)

# ---------------------------------------------------------------------------
# Immersed boundary machinery
# ---------------------------------------------------------------------------
ibmesh = IBMesh(0.0, Lx, 0.0, Ly, N, N, 2)
ib_interpolation = IBInterpolation(ibmesh)
coords_bg = Function(V)
coords_bg.interpolate(lambda x: np.array([x[0], x[1]]))
ibmesh.build_map(coords_bg._cpp_object)
ib_interpolation.evaluate_current_points(solid_coords._cpp_object)

# ---------------------------------------------------------------------------
# Probes and output
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

clamp_idx = np.where(np.isclose(ref_nodes[:, 0], XA))[0]

dx_fluid = Measure("dx", domain=mesh)
area_fluid = Lx * Ly

u_io = Function(V_io)
p_io = Function(Q)
f_io = Function(V_io)
if WRITE_XDMF:
    file_velocity = dolfinx.io.XDMFFile(comm, outdir + f"/velocity_{tag}.xdmf", "w")
    file_pressure = dolfinx.io.XDMFFile(comm, outdir + f"/pressure_{tag}.xdmf", "w")
    file_solid = dolfinx.io.XDMFFile(comm, outdir + f"/solid_{tag}.xdmf", "w")
    file_velocity.write_mesh(mesh)
    file_pressure.write_mesh(mesh)
    file_solid.write_mesh(structure)

csv_path = os.path.join(outdir, f"metrics_{tag}.csv")
with open(csv_path, "w") as fh:
    fh.write("step,t,corner_dY,max_disp,clamp_max_disp,J_min,J_max,uL2,traction\n")

log.set_log_level(log.LogLevel.WARNING)

def metrics(step, t):
    disp = solid_coords.x.array.reshape(-1, bs) - ref.reshape(-1, bs)
    mag = np.linalg.norm(disp, axis=1)
    J_func.interpolate(J_interp)
    jv = J_func.x.array
    uL2 = assemble_scalar(form(dot(ns_solver.u_, ns_solver.u_) * dx_fluid)) ** 0.5
    row = (step, t, solid_coords.x.array[probe_y] - ref[probe_y],
           mag.max(), mag[clamp_idx].max(),
           jv.min(), jv.max(), uL2, float(traction.value))
    with open(csv_path, "a") as fh:
        fh.write(",".join((str(row[0]),) + tuple(f"{v:.10g}" for v in row[1:]))
                 + "\n")
    if rank == 0:
        print(f"  step {step:7d} t={t:8.4f} dY={row[2]: .6f} "
              f"max|d|={row[3]:.4f} clamp={row[4]:.3e} "
              f"J=[{row[5]:.4f},{row[6]:.4f}]")
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
        row = metrics(step, t)
        if WRITE_XDMF and step % (OUT_EVERY * 10) == 0:
            u_io.interpolate(ns_solver.u_)
            p_io.interpolate(ns_solver.p_)
            solid_coords_io.interpolate(solid_coords)
            file_velocity.write_function(u_io, t)
            file_pressure.write_function(p_io, t)
            file_solid.write_function(solid_coords_io, t)

# ---------------------------------------------------------------------------
# Final snapshot for plotting
# ---------------------------------------------------------------------------
Xf = V.tabulate_dof_coordinates()[:, :2]
Uf = ns_solver.u_.x.array.reshape(-1, mesh.geometry.dim)
Xp = Q.tabulate_dof_coordinates()[:, :2]
Pp = ns_solver.p_.x.array
# array-order coordinates of the fluid velocity dofs (identity map), needed
# to pair column data with nodal values (tabulate rows are cell-ordered)
fluid_cx = Function(V)
fluid_cx.interpolate(lambda x: np.array([x[0], x[1]]))
# nodal vorticity field (the paper's spurious-vorticity diagnostic)
V_om = functionspace(mesh, element("Lagrange",
                                   mesh.topology.cell_name(), 1))
omega = Function(V_om, name="omega")
om_interp = Expression(
    ns_solver.u_[1].dx(0) - ns_solver.u_[0].dx(1),
    V_om.element.interpolation_points)
omega.interpolate(om_interp)
# structured (i, j) index of every solid node on the (2M+1) x (2MR+1) grid
t_nodes = (ref_nodes[:, 0] - XA) / (XB - XA)
yb_nodes = YA + t_nodes * (YB - YA)
yt_nodes = YD + t_nodes * (YC - YD)
solid_ij = np.column_stack([
    np.rint(t_nodes * M).astype(np.int64),
    np.rint((ref_nodes[:, 1] - yb_nodes) / (yt_nodes - yb_nodes)
            * MR).astype(np.int64)])
J_func.interpolate(J_interp)
np.savez_compressed(
    os.path.join(outdir, f"snapshot_{tag}.npz"),
    fluid_x=Xf, fluid_u=Uf, fluid_cx=fluid_cx.x.array.reshape(-1, 2),
    p_x=Xp, p=Pp,
    omega=omega.x.array, omega_x=V_om.tabulate_dof_coordinates()[:, :2],
    solid_x=ref_nodes, solid_coords=solid_coords.x.array.reshape(-1, bs),
    solid_ref=ref.reshape(-1, bs), solid_ij=solid_ij,
    J=J_func.x.array, J_x=VJ.tabulate_dof_coordinates()[:, :2],
    meta=np.array([M, MR, MFAC, N, DT, TL, TF, T_LOAD, float(BETA)]),
)
if rank == 0:
    print(f"[demo_441] done: {STEPS} steps, M={M} MFAC={MFAC} N={N}; "
          f"csv={csv_path}")
