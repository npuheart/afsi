"""
demo_403 — Channel flow over an elastic thick plate (beamInCrossFlow)
Section 4.5, Tuković et al. (2018)

3D immersed boundary FSI: fixed Cartesian fluid mesh + Lagrangian solid mesh.
"""

from petsc4py import PETSc
from mpi4py import MPI

import os
import numpy as np

import dolfinx
from dolfinx import log
from dolfinx.fem import (Function, functionspace,
                         dirichletbc, locate_dofs_topological)
from dolfinx.mesh import CellType, GhostMode, locate_entities, meshtags
from basix.ufl import element

from ufl import (Identity, SpatialCoordinate, TestFunction, inv, ln, det,
                 as_vector, dot, ds, dx, inner, grad, Measure)
from dolfinx.fem import form, assemble_scalar

from afsic import ChorinSolver, TimeManager
from afsic import swanlab_init, swanlab_upload
from dolfinx.fem.petsc import create_vector, assemble_vector
from configuration import config

swanlab_init(config['project_name'], config['experiment_name'], config,
             api_key="odR9FodGeQojOPlk2sir1")

# =============================================================================
# Fluid
# =============================================================================
mesh = dolfinx.mesh.create_box(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0, 0.0), (config["Lx"], config["Ly"], config["Lz"])),
    n=(config["Nx"], config["Ny"], config["Nz"]),
    cell_type=CellType.hexahedron,
    ghost_mode=GhostMode.shared_facet,
)

gdim = mesh.geometry.dim
fdim = mesh.topology.dim - 1

# Mark boundaries
marker_inlet = 1     # x = 0
marker_outlet = 2    # x = Lx
marker_bottom = 3    # y = 0 (no-slip wall)
marker_symmetry = 4  # y = Ly (symmetry plane)
marker_back = 5      # z = Lz (no-slip back wall)
marker_front = 6     # z = 0 (symmetry plane)

mesh.topology.create_connectivity(fdim, mesh.topology.dim)
boundaries = [
    (marker_inlet,    lambda x: np.isclose(x[0], 0)),
    (marker_outlet,   lambda x: np.isclose(x[0], config["Lx"])),
    (marker_bottom,   lambda x: np.isclose(x[1], 0)),
    (marker_symmetry, lambda x: np.isclose(x[1], config["Ly"])),
    (marker_back,     lambda x: np.isclose(x[2], config["Lz"])),
    (marker_front,    lambda x: np.isclose(x[2], 0)),
]

facet_indices, facet_markers = [], []
for (mk, loc) in boundaries:
    facets = locate_entities(mesh, fdim, loc)
    facet_indices.append(facets)
    facet_markers.append(np.full_like(facets, mk))
facet_indices = np.hstack(facet_indices).astype(np.int32)
facet_markers = np.hstack(facet_markers).astype(np.int32)
sf = np.argsort(facet_indices)
facet_tag = meshtags(mesh, fdim, facet_indices[sf], facet_markers[sf])

# Function spaces
v_cg2 = element("Lagrange", mesh.topology.cell_name(), 2,
                shape=(gdim,))
v_cg1 = element("Lagrange", mesh.topology.cell_name(), 1,
                shape=(gdim,))
s_cg1 = element("Lagrange", mesh.topology.cell_name(), 1)
V = functionspace(mesh, v_cg2)
V_io = functionspace(mesh, v_cg1)
Q = functionspace(mesh, s_cg1)


# ---- Parabolic inlet with smooth ramp ----
class InletVelocity:
    def __init__(self, Um, Ly, Lz, ramp_time):
        self.Um = Um
        self.Ly = Ly
        self.Lz = Lz
        self.ramp_time = ramp_time
        self.t = 0.0
        self.scale = 0.0

    def update(self, t):
        self.t = t
        if t < self.ramp_time:
            self.scale = self.Um * (1.0 - np.cos(np.pi * t / self.ramp_time)) / 2.0
        else:
            self.scale = self.Um

    def __call__(self, x):
        # Parabolic in y (height) and uniform in z (spanwise)
        values = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)
        H = self.Ly
        # u_x(y) = scale * 4 * y*(H-y) / H^2   (peak = scale at y=H/2)
        values[0] = self.scale * 4.0 * x[1] * (H - x[1]) / (H * H)
        return values


inlet_func = InletVelocity(config["Um"], config["Ly"], config["Lz"],
                           config["ramp_time"])
u_inlet = Function(V)
u_inlet.interpolate(inlet_func)
bcu_inlet = dirichletbc(u_inlet, locate_dofs_topological(
    V, fdim, facet_tag.find(marker_inlet)))

# No-slip on bottom and back wall; slip on symmetry planes
u_zero = Function(V)
u_zero.x.array[:] = 0.0

# y=0 (bottom wall): no-slip (all components zero)
bcu_bottom = dirichletbc(u_zero, locate_dofs_topological(
    V, fdim, facet_tag.find(marker_bottom)))

# z=Lz (back wall): no-slip
bcu_back = dirichletbc(u_zero, locate_dofs_topological(
    V, fdim, facet_tag.find(marker_back)))

# Symmetry at y=Ly: v_y = 0 (slip in x,z), use dirichlet on y-component only
# For simplicity, we treat it as slip by not constraining x,z components.
# Symmetry at z=0: v_z = 0 (slip in x,y).

bcu = [bcu_inlet, bcu_bottom, bcu_back]

# Pressure outlet: p = 0 at x = Lx
bcp_outlet = dirichletbc(PETSc.ScalarType(0.0),
                         locate_dofs_topological(Q, fdim,
                                                  facet_tag.find(marker_outlet)),
                         Q)
bcp = [bcp_outlet]

# Solver
ns_solver = ChorinSolver(V, Q, bcu, bcp, config['dt'], config['rho'],
                         config['mu'])

# =============================================================================
# Solid
# =============================================================================
plate_path = os.path.join(os.path.dirname(__file__), "plate_mesh.xdmf")
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, plate_path, "r") as xdmf:
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

dVs = TestFunction(Vs)
mu_s = config["mu_s"]
lambda_s = config["lambda_s"]
beta = config["beta"]

FF = grad(solid_coords)
J = det(FF)

# Penalty: fix plate bottom face (y = 0) to its initial position
X0 = SpatialCoordinate(structure)
structure.topology.create_connectivity(
    structure.topology.dim - 1, structure.topology.dim)
bottom_facets = locate_entities(structure, fdim,
                                lambda x: np.isclose(x[1], 0.0))
bottom_marker = 10
facet_indices_s = [bottom_facets]
facet_markers_s = [np.full_like(bottom_facets, bottom_marker)]
facet_indices_s = np.hstack(facet_indices_s).astype(np.int32)
facet_markers_s = np.hstack(facet_markers_s).astype(np.int32)
sf_s = np.argsort(facet_indices_s)
facet_tag_s = meshtags(structure, fdim,
                       facet_indices_s[sf_s], facet_markers_s[sf_s])

dss = Measure("ds", domain=structure, subdomain_data=facet_tag_s)
dxx = Measure("dx", domain=structure)

# Constraint: fix bottom plate position in all directions
constraint = solid_coords - X0

# Neo-Hookean PK1 stress: P = mu_s*(F - F^{-T}) + lambda_s*ln(J)*F^{-T}
I1 = inner(FF, FF)
P_iso = mu_s * J**(-2.0 / gdim) * (FF - (I1 / gdim) * inv(FF).T)
P_vol = lambda_s * ln(J) * inv(FF).T
P_s = P_iso + P_vol

# Variational form: elastic + penalty on bottom face
L_hat = form(
    -inner(P_s, grad(dVs)) * dxx
    - beta * inner(constraint, dVs) * dss(bottom_marker)
)
b1 = create_vector(Vs)  # dolfinx 0.10.0: create_vector 需函数空间而非 Form

# =============================================================================
# Immersed Boundary coupling
# =============================================================================
from afsic import IBMesh3D, IBInterpolation3D

ibmesh = IBMesh3D(0.0, config["Lx"], 0.0, config["Ly"], 0.0, config["Lz"],
                  config["Nx"], config["Ny"], config["Nz"],
                  config["velocity_order"])
ib_interpolation = IBInterpolation3D(ibmesh)

coords_bg = Function(V)
coords_bg.interpolate(lambda x: np.array([x[0], x[1], x[2]]))
solid_coords.interpolate(lambda x: np.array([x[0], x[1], x[2]]))
ibmesh.build_map(coords_bg._cpp_object)
ib_interpolation.evaluate_current_points(solid_coords._cpp_object)

# =============================================================================
# Output
# =============================================================================
u_io = Function(V_io)
p_io = Function(Q)
file_velocity = dolfinx.io.XDMFFile(mesh.comm,
                                     config["output_path"] + "velocity.xdmf", "w")
file_pressure = dolfinx.io.XDMFFile(mesh.comm,
                                     config["output_path"] + "pressure.xdmf", "w")
file_solid = dolfinx.io.XDMFFile(mesh.comm,
                                  config["output_path"] + "solid.xdmf", "w")
file_velocity.write_mesh(mesh)
file_pressure.write_mesh(mesh)
file_solid.write_mesh(structure)

time_manager = TimeManager(config['T'], config['num_steps'], fps=20)

form_u_L2 = form(dot(ns_solver.u_, ns_solver.u_) * dx)
form_p_L2 = form(dot(ns_solver.p_, ns_solver.p_) * dx)
form_F_L2 = form(dot(solid_coords, solid_coords) * dx)

log.set_log_level(log.LogLevel.INFO)

# =============================================================================
# Time loop
# =============================================================================
for step in range(config['num_steps']):
    current_time = step * config['dt']

    # Update inlet velocity (ramp)
    inlet_func.update(current_time)
    u_inlet.interpolate(inlet_func)

    # Solve fluid
    ns_solver.solve_one_step()

    # Fluid -> Solid: interpolate velocity
    ib_interpolation.fluid_to_solid(ns_solver.u_._cpp_object,
                                    solid_velocity._cpp_object)

    # Update solid position (explicit Euler)
    solid_coords.x.array[:] += solid_velocity.x.array[:] * config['dt']
    solid_coords.x.scatter_forward()

    # Diagnostics
    u_L2 = mesh.comm.allreduce(assemble_scalar(form_u_L2), op=MPI.SUM)
    p_L2 = mesh.comm.allreduce(assemble_scalar(form_p_L2), op=MPI.SUM)
    F_L2 = mesh.comm.allreduce(assemble_scalar(form_F_L2), op=MPI.SUM)

    # Update IB points
    ib_interpolation.evaluate_current_points(solid_coords._cpp_object)

    # Compute solid force
    with b1.localForm() as loc:
        loc.set(0)
    assemble_vector(b1, L_hat)
    b1.ghostUpdate(addv=PETSc.InsertMode.ADD,
                   mode=PETSc.ScatterMode.REVERSE)
    with b1.getBuffer() as arr:
        solid_force.x.array[:len(arr)] = arr[:]

    # Solid -> Fluid: spread force
    ib_interpolation.solid_to_fluid(ns_solver.f._cpp_object,
                                    solid_force._cpp_object)
    ns_solver.f.x.scatter_forward()

    # Output
    data_log = {}
    if time_manager.should_output(step):
        u_io.interpolate(ns_solver.u_)
        p_io.interpolate(ns_solver.p_)
        file_velocity.write_function(u_io, current_time)
        file_pressure.write_function(p_io, current_time)
        solid_force_io.interpolate(solid_force)
        solid_coords_io.interpolate(solid_coords)
        file_solid.write_function(solid_force_io, current_time)
        file_solid.write_function(solid_coords_io, current_time)
        if MPI.COMM_WORLD.rank == 0:
            data_log["u_norm"] = u_L2
            data_log["p_norm"] = p_L2
            data_log["solid_force_norm"] = F_L2
            data_log["inlet_velocity"] = inlet_func.scale
            print(f"Step {step+1}/{config['num_steps']}, "
                  f"Time: {current_time:.3f}s, "
                  f"U_in: {inlet_func.scale:.1f} cm/s")
            swanlab_upload(current_time, data_log)

if MPI.COMM_WORLD.rank == 0:
    print("Simulation complete.")
