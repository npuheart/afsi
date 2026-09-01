"""Static equilibrium of an immersed anisotropic annular solid.

This is a port/adaptation of the benchmark in the description:
  - 2D unit square [0,1]^2, no-slip on all four walls;
  - fluid: rho_f = 1.0, mu_f = 1.0;
  - annular solid: R = 0.25, w = 0.0625, center (0.5, 0.5);
  - solid material: S^s = mu_s e_theta (x) e_theta, mu_s = 1.0;
  - default dt = 1e-4, 100 steps, zero-mean pressure;
    use DT=1e-3 STEPS=10 to reproduce the paper's time-stepping setting.

The implementation follows the existing afsic demos (Chorin + IBMesh/
IBInterpolation).  A point pressure Dirichlet BC is used only to make the
pressure Poisson problem non-singular; after the solve the pressure is shifted
to have zero mean, which is the gauge used by the analytical solution.

Note on force scaling
---------------------
The current afsic `IBInterpolation.solid_to_fluid` spreads the assembled FE
force vector as point forces with a fixed Lagrangian weight w = 1.  For this
static pressure benchmark this makes the resulting Eulerian body-force field
one half of the continuum IB force.  We therefore multiply the assembled solid
force by `force_scale = 1.0` (default).  An earlier version of this demo used
a factor ~2 because the custom quadrilateral mesh used the wrong DOLFINx vertex
ordering, which corrupted the solid FE integrals.  After fixing the mesh
ordering no empirical force scaling is needed.
"""
import os
from mpi4py import MPI
from petsc4py import PETSc

import numpy as np

import dolfinx
from dolfinx import log
from dolfinx.fem import (Function, functionspace,
                         dirichletbc, locate_dofs_topological, form,
                         assemble_scalar, Constant)
from dolfinx.mesh import CellType, GhostMode, locate_entities, meshtags
from basix.ufl import element
from dolfinx.fem.petsc import create_vector, assemble_vector

from ufl import (Measure, TestFunction, SpatialCoordinate,
                 as_vector, conditional, And, Or, lt, gt, dot, dx, inner,
                 grad, sqrt, pi)

from afsic import ChorinSolver, IPCSSolver, TimeManager, IBMesh, IBInterpolation
from materials import CircumferentialMaterial

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N = int(os.environ.get("N", "32"))        # fluid elements along one side
STEPS = int(os.environ.get("STEPS", "100"))
DT = float(os.environ.get("DT", "1.0e-4"))
FORCE_SCALE = float(os.environ.get("FORCE_SCALE", "1.0"))
SOLVER = os.environ.get("SOLVER", "chorin").lower()

R = 0.25          # inner radius
w = 0.0625        # annulus width
mu_s = 1.0        # circumferential modulus
rho_f = 1.0
mu_f = 1.0
dt = DT
Lx = Ly = 1.0

config = {
    "nssolver": "chorinsolver",
    "project_name": "demo-423",
    "tag": "annular-static",
    "velocity_order": 2,
    "force_order": 2,
    "pressure_order": 1,
    "num_processors": MPI.COMM_WORLD.size,
    "T": STEPS * dt,
    "dt": dt,
    "rho": rho_f,
    "mu": mu_f,
    "Lx": Lx,
    "Ly": Ly,
    "Nx": N,
    "Ny": N,
    "mu_s": mu_s,
    "R": R,
    "w": w,
    "force_scale": FORCE_SCALE,
}
config["num_steps"] = STEPS

_demo_dir = os.path.dirname(os.path.abspath(__file__))
config["output_path"] = os.path.join(_demo_dir, "plot") + os.sep
os.makedirs(config["output_path"], exist_ok=True)

# ---------------------------------------------------------------------------
# Fluid mesh and boundary conditions
# ---------------------------------------------------------------------------
mesh = dolfinx.mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (Lx, Ly)),
    n=(N, N),
    cell_type=CellType.quadrilateral,
    ghost_mode=GhostMode.shared_facet,
)

mesh.topology.create_connectivity(1, 2)
marker_left, marker_right, marker_down, marker_up = 1, 2, 3, 4
boundaries = [
    (marker_left,  lambda x: np.isclose(x[0], 0)),
    (marker_right, lambda x: np.isclose(x[0], Lx)),
    (marker_down,  lambda x: np.isclose(x[1], 0)),
    (marker_up,    lambda x: np.isclose(x[1], Ly)),
]

fdim = mesh.topology.dim - 1
facet_indices, facet_markers = [], []
for marker, locator in boundaries:
    facets = locate_entities(mesh, fdim, locator)
    facet_indices.append(facets)
    facet_markers.append(np.full_like(facets, marker))
facet_indices = np.hstack(facet_indices).astype(np.int32)
facet_markers = np.hstack(facet_markers).astype(np.int32)
sorted_facets = np.argsort(facet_indices)
facet_tag = meshtags(mesh, fdim, facet_indices[sorted_facets],
                     facet_markers[sorted_facets])

v_cg2 = element("Lagrange", mesh.topology.cell_name(),
                2, shape=(mesh.geometry.dim,))
v_cg1 = element("Lagrange", mesh.topology.cell_name(),
                1, shape=(mesh.geometry.dim,))
s_cg1 = element("Lagrange", mesh.topology.cell_name(), 1)
V = functionspace(mesh, v_cg2)
Q = functionspace(mesh, s_cg1)
V_io = functionspace(mesh, v_cg1)

# No-slip on all walls
u_zero = np.array((0,) * mesh.geometry.dim, dtype=PETSc.ScalarType)
bcu = [dirichletbc(u_zero, locate_dofs_topological(V, fdim, facet_tag.find(i)), V)
       for i in (marker_left, marker_right, marker_down, marker_up)]

# Pressure gauge: point value to remove the null space; zero-mean is applied
# afterwards for the error computation.
point_loc = dolfinx.mesh.locate_entities_boundary(
    mesh, 0, lambda x: np.logical_and(np.isclose(x[0], 0),
                                      np.isclose(x[1], 0)))
bcp = [dirichletbc(PETSc.ScalarType(0.0),
                   locate_dofs_topological(Q, 0, point_loc), Q)]

if SOLVER == "ipcs":
    ns_solver = IPCSSolver(V, Q, bcu, bcp, config["dt"], config["rho"], config["mu"])
else:
    ns_solver = ChorinSolver(V, Q, bcu, bcp, config["dt"], config["rho"], config["mu"])

# ---------------------------------------------------------------------------
# Solid structure
# ---------------------------------------------------------------------------
mesh_path = os.path.join(_demo_dir, "plot", "mesh-423.xdmf")
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, mesh_path, "r") as xdmf:
    structure = xdmf.read_mesh(name="mesh")

v_cg2_s = element("Lagrange", structure.topology.cell_name(),
                  config["force_order"], shape=(structure.geometry.dim,))
v_cg1_s = element("Lagrange", structure.topology.cell_name(),
                  1, shape=(structure.geometry.dim,))
Vs = functionspace(structure, v_cg2_s)
Vs_io = functionspace(structure, v_cg1_s)

solid_coords = Function(Vs, name="solid_coords")
solid_coords_io = Function(Vs_io, name="solid_coords_io")
solid_force = Function(Vs, name="solid_force")
solid_force_io = Function(Vs_io, name="solid_force_io")
solid_velocity = Function(Vs, name="solid_velocity")

# Material and weak form: internal virtual work -> nodal force vector.
dVs = TestFunction(Vs)
material = CircumferentialMaterial(mu_s=config["mu_s"], center=(0.5, 0.5))
PK1 = material.first_piola_kirchhoff_stress_v1(structure, solid_coords)
L_hat = form(-inner(PK1, grad(dVs)) * dx)
b1 = create_vector(Vs)

# ---------------------------------------------------------------------------
# Immersed-boundary interpolation
# ---------------------------------------------------------------------------
ibmesh = IBMesh(0.0, config["Lx"], 0.0, config["Ly"],
                config["Nx"], config["Ny"], config["velocity_order"])
ib_interpolation = IBInterpolation(ibmesh)
coords_bg = Function(V)
coords_bg.interpolate(lambda x: np.array([x[0], x[1]]))
solid_coords.interpolate(lambda x: np.array([x[0], x[1]]))
ibmesh.build_map(coords_bg._cpp_object)
ib_interpolation.evaluate_current_points(solid_coords._cpp_object)

# ---------------------------------------------------------------------------
# Output / error machinery
# ---------------------------------------------------------------------------
u_io = Function(V_io)
p_io = Function(Q)
f_io = Function(V_io)
file_velocity = dolfinx.io.XDMFFile(
    mesh.comm, config["output_path"] + "velocity.xdmf", "w")
file_pressure = dolfinx.io.XDMFFile(
    mesh.comm, config["output_path"] + "pressure.xdmf", "w")
file_solid = dolfinx.io.XDMFFile(
    mesh.comm, config["output_path"] + "solid_force.xdmf", "w")
file_fluid_force = dolfinx.io.XDMFFile(
    mesh.comm, config["output_path"] + "fluid_force.xdmf", "w")
file_velocity.write_mesh(mesh)
file_pressure.write_mesh(mesh)
file_fluid_force.write_mesh(mesh)
file_solid.write_mesh(structure)

time_manager = TimeManager(config["T"], config["num_steps"], fps=10)
dx_fluid = Measure("dx", domain=mesh)

# Analytical pressure (zero mean)
def exact_pressure(x):
    rr = np.sqrt((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2)
    const = -np.pi * config["mu_s"] / (2.0 * config["Lx"] * config["Ly"]) * (
        (R + w) ** 2 - R ** 2
    )
    tiny = 1.0e-14
    inside = config["mu_s"] * np.log(1.0 + w / R) + const
    ring = config["mu_s"] * np.log((R + w) / np.maximum(rr, tiny)) + const
    outside = const
    return np.where(rr <= R, inside, np.where(rr < R + w, ring, outside))

p_exact = Function(Q)
p_exact.interpolate(exact_pressure)

area = assemble_scalar(form(Constant(mesh, PETSc.ScalarType(1.0)) * dx_fluid))

# ---------------------------------------------------------------------------
# Time loop
# ---------------------------------------------------------------------------
log.set_log_level(log.LogLevel.INFO)
for step in range(config["num_steps"]):
    current_time = step * config["dt"]

    ns_solver.solve_one_step()

    # Advect the Lagrangian solid with the interpolated fluid velocity
    ib_interpolation.fluid_to_solid(ns_solver.u_._cpp_object,
                                    solid_velocity._cpp_object)
    solid_coords.x.array[:] += solid_velocity.x.array[:] * config["dt"]
    solid_coords.x.scatter_forward()

    # Re-evaluate current solid point locations and compute solid force
    ib_interpolation.evaluate_current_points(solid_coords._cpp_object)
    with b1.localForm() as loc:
        loc.set(0)
    assemble_vector(b1, L_hat)
    b1.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    with b1.getBuffer() as arr:
        # Nodal IB spreading: no separate w is used.  The factor is an
        # empirical calibration for this static pressure benchmark.
        solid_force.x.array[: len(arr)] = config["force_scale"] * arr[:]

    ib_interpolation.solid_to_fluid(ns_solver.f._cpp_object,
                                    solid_force._cpp_object)
    ns_solver.f.x.scatter_forward()

    if time_manager.should_output(step):
        u_io.interpolate(ns_solver.u_)
        p_io.interpolate(ns_solver.p_)
        f_io.interpolate(ns_solver.f)
        solid_force_io.interpolate(solid_force)
        solid_coords_io.interpolate(solid_coords)
        file_velocity.write_function(u_io, current_time)
        file_pressure.write_function(p_io, current_time)
        file_fluid_force.write_function(f_io, current_time)
        file_solid.write_function(solid_force_io, current_time)
        file_solid.write_function(solid_coords_io, current_time)

# ---------------------------------------------------------------------------
# Final errors
# ---------------------------------------------------------------------------
# Shift numerical pressure to zero mean (same gauge as the analytical solution)
mean_p = assemble_scalar(form(ns_solver.p_ * dx_fluid)) / area
p_shift = Function(Q)
p_shift.x.array[:] = ns_solver.p_.x.array[:] - mean_p

# Also shift the exact pressure to the same discrete zero-mean gauge
mean_p_exact = assemble_scalar(form(p_exact * dx_fluid)) / area
p_exact_shift = Function(Q)
p_exact_shift.x.array[:] = p_exact.x.array[:] - mean_p_exact

e_v_L2 = assemble_scalar(form(dot(ns_solver.u_, ns_solver.u_) * dx_fluid)) ** 0.5
e_v_H1 = assemble_scalar(
    form((dot(ns_solver.u_, ns_solver.u_) +
          inner(grad(ns_solver.u_), grad(ns_solver.u_))) * dx_fluid)) ** 0.5
e_p_L2 = assemble_scalar(
    form((p_shift - p_exact_shift) ** 2 * dx_fluid)) ** 0.5

# ---------------------------------------------------------------------------
# Output error fields (final time)
# ---------------------------------------------------------------------------
u_err = Function(V_io, name="velocity_error")
u_err.interpolate(ns_solver.u_)  # exact velocity is zero

p_err = Function(Q, name="pressure_error")
p_err.x.array[:] = p_shift.x.array[:] - p_exact_shift.x.array[:]

file_velocity_error = dolfinx.io.XDMFFile(
    mesh.comm, config["output_path"] + "velocity_error.xdmf", "w")
file_pressure_error = dolfinx.io.XDMFFile(
    mesh.comm, config["output_path"] + "pressure_error.xdmf", "w")
file_velocity_error.write_mesh(mesh)
file_pressure_error.write_mesh(mesh)
file_velocity_error.write_function(u_err, config["T"])
file_pressure_error.write_function(p_err, config["T"])
file_velocity_error.close()
file_pressure_error.close()
file_fluid_force.close()

# Split pressure error: inner disk vs interface smear band
Xf = SpatialCoordinate(mesh)
rf = sqrt((Xf[0] - 0.5) ** 2 + (Xf[1] - 0.5) ** 2)
h = 1.0 / N
band_width = 2.0 * h
inside_mask = conditional(rf <= R - band_width, 1.0, 0.0)
inside4_mask = conditional(rf <= R - 4.0 * band_width, 1.0, 0.0)
band_mask = conditional(
    Or(And(gt(rf, R - band_width), lt(rf, R + band_width)),
       And(gt(rf, R + w - band_width), lt(rf, R + w + band_width))),
    1.0, 0.0)
outer_mask = conditional(rf >= R + w + band_width, 1.0, 0.0)
e_p_inner = assemble_scalar(
    form(((p_shift - p_exact_shift) ** 2) * inside_mask * dx_fluid)) ** 0.5
e_p_inner4 = assemble_scalar(
    form(((p_shift - p_exact_shift) ** 2) * inside4_mask * dx_fluid)) ** 0.5
area_inner4 = assemble_scalar(form(inside4_mask * dx_fluid))
rms_inner4 = e_p_inner4 / sqrt(area_inner4) if area_inner4 > 0 else 0.0
e_p_band = assemble_scalar(
    form(((p_shift - p_exact_shift) ** 2) * band_mask * dx_fluid)) ** 0.5
e_p_outer = assemble_scalar(
    form(((p_shift - p_exact_shift) ** 2) * outer_mask * dx_fluid)) ** 0.5

# Region areas and RMS values, plus discrete means for gauge check
area_inner = assemble_scalar(form(inside_mask * dx_fluid))
area_band = assemble_scalar(form(band_mask * dx_fluid))
area_outer = assemble_scalar(form(outer_mask * dx_fluid))
rms_inner = e_p_inner / sqrt(area_inner)
rms_band = e_p_band / sqrt(area_band)
rms_outer = e_p_outer / sqrt(area_outer)
mean_p_shift = assemble_scalar(form(p_shift * dx_fluid)) / area

if MPI.COMM_WORLD.rank == 0:
    print("=" * 60)
    print(f"N={N}, M={2 * N // 16}, force_scale={config['force_scale']}")
    print(f"Steps={config['num_steps']}, dt={config['dt']}, "
          f"T={config['T']}")
    print(f"e_v_L2 = {e_v_L2:.6e}")
    print(f"e_v_H1 = {e_v_H1:.6e}")
    print(f"e_p_L2 = {e_p_L2:.6e}")
    print(f"max|v| = {np.max(np.abs(ns_solver.u_.x.array)):.6e}")
    print(f"p range (zero mean): {p_shift.x.array.min():.6e}, "
          f"{p_shift.x.array.max():.6e}")
    print(f"e_p inner(r<R-2h) = {e_p_inner:.6e}")
    print(f"e_p inner4(r<R-4h) = {e_p_inner4:.6e}, RMS = {rms_inner4:.6e}")
    print(f"e_p band(|r-R|<2h) = {e_p_band:.6e}")
    print(f"e_p outer(r>R+w+2h) = {e_p_outer:.6e}")
    print(f"area inner/band/outer = {area_inner:.6e}/{area_band:.6e}/{area_outer:.6e}")
    print(f"RMS inner/band/outer = {rms_inner:.6e}/{rms_band:.6e}/{rms_outer:.6e}")
    print(f"mean p_shift = {mean_p_shift:.6e}, mean p_exact = {mean_p_exact:.6e}")
    if os.environ.get("SAMPLE"):
        from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
        tree = bb_tree(mesh, mesh.geometry.dim)
        print(" x       p_num      p_exact")
        for xs in [0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9]:
            x0 = np.array([xs, 0.5, 0.0], dtype=dolfinx.default_scalar_type)
            cells = compute_colliding_cells(mesh, compute_collisions_points(tree, x0), x0)
            if len(cells.array) == 0:
                continue
            cell = cells.array[0]
            pn = p_shift.eval(x0, cell)[0]
            pe = p_exact_shift.eval(x0, cell)[0]
            print(f"{xs:5.2f} {pn: .6e} {pe: .6e}")
    if os.environ.get("PROFILE"):
        from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
        tree = bb_tree(mesh, mesh.geometry.dim)
        xs = np.linspace(0.5, 0.9, 801)
        rows = []
        for xv in xs:
            x0 = np.array([xv, 0.5, 0.0], dtype=dolfinx.default_scalar_type)
            cells = compute_colliding_cells(mesh, compute_collisions_points(tree, x0), x0)
            if len(cells.array) == 0:
                pn = np.nan
                pe = np.nan
            else:
                cell = cells.array[0]
                pn = p_shift.eval(x0, cell)[0]
                pe = p_exact_shift.eval(x0, cell)[0]
            rows.append((xv, pn, pe, pn - pe))
        if MPI.COMM_WORLD.rank == 0:
            out = os.path.join(config["output_path"], f"profile_N{N}.csv")
            np.savetxt(out, np.array(rows), header="x,pnum,pexact,err", delimiter=",")
            print(f"Profile written to {out}")
    if os.environ.get("CIRCLE"):
        from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
        tree = bb_tree(mesh, mesh.geometry.dim)
        thetas = np.linspace(0.0, 2.0 * np.pi, 721)
        errs = []
        for th in thetas:
            x0 = np.array([0.5 + R * np.cos(th), 0.5 + R * np.sin(th), 0.0],
                          dtype=dolfinx.default_scalar_type)
            cells = compute_colliding_cells(mesh, compute_collisions_points(tree, x0), x0)
            if len(cells.array) == 0:
                continue
            cell = cells.array[0]
            pn = p_shift.eval(x0, cell)[0]
            pe = p_exact_shift.eval(x0, cell)[0]
            errs.append(abs(pn - pe))
        if MPI.COMM_WORLD.rank == 0 and errs:
            print(f"inner circle max|err|={max(errs):.6e} mean|err|={np.mean(errs):.6e} std={np.std(errs):.6e}")
    print("=" * 60)
