"""Flow past a rigid cylinder — IB-FE method.

Uses the standard Peskin Immersed Boundary feedback force coupling
(same as demo_402 Turek FSI), with the cylinder modelled as a
Neo-Hookean solid disk of extremely high stiffness.

Reference:
    Ma, Cai, Wang & Gao (2025) "AFSI: Automated Fluid-Structure
    Interaction Solver Development for Nonlinear Solid Mechanics."
    arXiv:2509.00014.

Method (per time step):
  1. Solve NS with current body force f
  2. Interpolate fluid velocity to solid mesh nodes
  3. Update solid position: X += v * dt
  4. Compute solid internal forces via Neo-Hookean stress
  5. Spread solid forces back to fluid grid as f

The cylinder is made effectively rigid by using a very large
shear modulus (mu_s ≈ 10^7 × factor).

Run:
    1. python generate_mesh.py    # generates cylinder_solid.xdmf
    2. python main.py             # or  mpirun -n <N> python main.py
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

from ufl import (FacetNormal, Identity, Measure, SpatialCoordinate,
                 TestFunction, TrialFunction, inv, ln, det,
                 as_vector, dot, ds, dx, inner, lhs, grad, nabla_grad,
                 rhs, sym, system)
from dolfinx.fem import form, assemble_scalar
from dolfinx.fem.petsc import create_vector, assemble_vector

from afsic import ChorinSolver, TimeManager
from afsic import swanlab_init, swanlab_upload
from configuration import config

swanlab_init(config['project_name'], config['experiment_name'], config,
             api_key="odR9FodGeQojOPlk2sir1")

# ==========================================================================
# 1. Fluid mesh — full rectangle
# ==========================================================================
mesh = dolfinx.mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (config["Lx"], config["Ly"])),
    n=(config["Nx"], config["Ny"]),
    cell_type=CellType.quadrilateral,
    ghost_mode=GhostMode.shared_facet,
)

mesh.topology.create_connectivity(1, 2)
marker_inlet, marker_outlet, marker_bottom, marker_top = 14, 12, 11, 13
boundaries = [
    (14, lambda x: np.isclose(x[0], 0)),
    (12, lambda x: np.isclose(x[0], config["Lx"])),
    (11, lambda x: np.isclose(x[1], 0)),
    (13, lambda x: np.isclose(x[1], config["Ly"])),
]

facet_indices, facet_markers = [], []
fdim = mesh.topology.dim - 1
for (marker, locator) in boundaries:
    facets = locate_entities(mesh, fdim, locator)
    facet_indices.append(facets)
    facet_markers.append(np.full_like(facets, marker))
facet_indices = np.hstack(facet_indices).astype(np.int32)
facet_markers = np.hstack(facet_markers).astype(np.int32)
sorted_facets = np.argsort(facet_indices)
facet_tag = meshtags(
    mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

v_cg2 = element("Lagrange", mesh.topology.cell_name(),
                2, shape=(mesh.geometry.dim,))
v_cg1 = element("Lagrange", mesh.topology.cell_name(),
                1, shape=(mesh.geometry.dim,))
s_cg1 = element("Lagrange", mesh.topology.cell_name(), 1)
V_io = functionspace(mesh, v_cg1)
V    = functionspace(mesh, v_cg2)
Q    = functionspace(mesh, s_cg1)

fdim = mesh.topology.dim - 1
gdim = mesh.geometry.dim


class InletVelocity:
    """Parabolic inlet: u_x(y) = 1.5*Um*y*(Ly-y)/(Ly/2)^2, 2s ramp."""
    def __init__(self, Um, Ly):
        self.t = 0.0
        self.t_ramp = 2.0
        self.Um = Um
        self.Ly = Ly
        self.scale = 0.0

    def update(self, t):
        self.t = t
        if self.t < self.t_ramp:
            self.scale = self.Um * (
                1.0 - np.cos(np.pi * self.t / self.t_ramp)) / 2.0
        else:
            self.scale = self.Um

    def __call__(self, x):
        values = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)
        H = self.Ly
        values[0] = 1.5 * self.scale * x[1] * (H - x[1]) / (H / 2.0) ** 2
        return values


inlet_velocity = InletVelocity(config["Um"], config["Ly"])
u_inlet_func = Function(V)
u_inlet_func.interpolate(inlet_velocity)
bcu_inlet = dirichletbc(
    u_inlet_func,
    locate_dofs_topological(V, fdim, facet_tag.find(marker_inlet)))

u_zero = Function(V)
u_zero.x.array[:] = 0.0
bcu_bottom = dirichletbc(
    u_zero, locate_dofs_topological(V, fdim, facet_tag.find(marker_bottom)))
bcu_top = dirichletbc(
    u_zero, locate_dofs_topological(V, fdim, facet_tag.find(marker_top)))

bcp_outlet = dirichletbc(
    PETSc.ScalarType(0.0),
    locate_dofs_topological(Q, fdim, facet_tag.find(marker_outlet)), Q)

bcu = [bcu_inlet, bcu_bottom, bcu_top]
bcp = [bcp_outlet]

ns_solver = ChorinSolver(V, Q, bcu, bcp,
                         config['dt'], config['rho'], config['mu'])

# ==========================================================================
# 2. Solid — Neo-Hookean disk (rigid via high stiffness + penalty)
# ==========================================================================
solid_mesh_path = os.path.join(os.path.dirname(__file__),
                                "./cylinder_solid.xdmf")
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, solid_mesh_path, "r") as xdmf:
    structure = xdmf.read_mesh(name="mesh")
    structure.topology.create_connectivity(
        structure.topology.dim, structure.topology.dim - 1)
    cell_tags = xdmf.read_meshtags(structure, name="cell_tags")
    facet_tags = xdmf.read_meshtags(structure, name="facet_tags")

# Both fluid and solid are in SI [m] — no scaling needed
# (demo_402 scaled to cm, but we stay in SI)

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

# ---- Neo-Hookean material ----
dVs = TestFunction(Vs)
mu_s = config["mu_s"]
lambda_s = config["lambda_s"]
beta = config["beta"]

FF = grad(solid_coords)
J = det(FF)

# Fix the ENTIRE disk via penalty on cell tag 1
X0 = SpatialCoordinate(structure)
dxx = Measure("dx", domain=structure, subdomain_data=cell_tags)
x_constraint = solid_coords[0] - X0[0]
y_constraint = solid_coords[1] - X0[1]
constraint = as_vector((x_constraint, y_constraint))

# Neo-Hookean: P = mu_s*(F - F^-T) + lambda_s*ln(J)*F^-T
I1 = inner(FF, FF)
P_iso = mu_s * J**(-2.0 / 2.0) * (FF - (I1 / 2.0) * inv(FF).T)
P_vol = lambda_s * ln(J) * inv(FF).T
P_s = P_iso + P_vol

# Weak form: internal stress + penalty fixation on whole disk
L_hat = form(
    -inner(P_s, grad(dVs)) * dxx(1)
    - beta * inner(constraint, dVs) * dxx(1)
)
b1 = create_vector(L_hat)

# ==========================================================================
# 3. IBM coupling — same as demo_402
# ==========================================================================
from afsic import IBMesh, IBInterpolation

ibmesh = IBMesh(0.0, config["Lx"],
                0.0, config["Ly"],
                config["Nx"], config["Ny"],
                config["velocity_order"])
ib_interpolation = IBInterpolation(ibmesh)

coords_bg = Function(V)
coords_bg.interpolate(lambda x: np.array([x[0], x[1]]))
solid_coords.interpolate(lambda x: np.array([x[0], x[1]]))
ibmesh.build_map(coords_bg._cpp_object)
ib_interpolation.evaluate_current_points(solid_coords._cpp_object)

# ==========================================================================
# 4. Output
# ==========================================================================
u_io = Function(V_io)
p_io = Function(Q)
file_vel = dolfinx.io.XDMFFile(
    mesh.comm, config["output_path"] + "velocity.xdmf", "w")
file_pre = dolfinx.io.XDMFFile(
    mesh.comm, config["output_path"] + "pressure.xdmf", "w")
file_sld = dolfinx.io.XDMFFile(
    mesh.comm, config["output_path"] + "solid.xdmf", "w")
file_vel.write_mesh(mesh)
file_pre.write_mesh(mesh)
file_sld.write_mesh(structure)

time_manager = TimeManager(config['T'], config['num_steps'], fps=100)

form_u_L2 = form(dot(ns_solver.u_, ns_solver.u_) * dx)
form_p_L2 = form(dot(ns_solver.p_, ns_solver.p_) * dx)
form_F_L2 = form(dot(solid_coords, solid_coords) * dx)

log.set_log_level(log.LogLevel.INFO)

if MPI.COMM_WORLD.rank == 0:
    D = 2 * 0.05  # cylinder diameter [m]
    Re = config['rho'] * config['Um'] * D / config['mu']
    print(f"Cylinder IB-FE: mu_s={mu_s:.1e}, lambda_s={lambda_s:.1e}, "
          f"beta={beta:.1e}")
    print(f"Fluid: {config['Nx']}×{config['Ny']}, "
          f"dt={config['dt']}, Re≈{Re:.0f}")

# ==========================================================================
# 5. Time loop — standard IB-FE feedback coupling
# ==========================================================================
for step in range(config['num_steps']):
    current_time = step * config['dt']
    inlet_velocity.update(current_time)
    u_inlet_func.interpolate(inlet_velocity)

    # -- Solve fluid --
    ns_solver.solve_one_step()

    # -- Interpolate fluid velocity → solid --
    ib_interpolation.fluid_to_solid(
        ns_solver.u_._cpp_object, solid_velocity._cpp_object)

    # -- Update solid position (forward Euler) --
    solid_coords.x.array[:] += solid_velocity.x.array[:] * config['dt']
    solid_coords.x.scatter_forward()

    # -- Compute solid internal force (Neo-Hookean + penalty) --
    ib_interpolation.evaluate_current_points(solid_coords._cpp_object)
    with b1.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b1, L_hat)
    b1.ghostUpdate(addv=PETSc.InsertMode.ADD,
                   mode=PETSc.ScatterMode.REVERSE)
    with b1.getBuffer() as arr:
        solid_force.x.array[:len(arr)] = arr[:]

    # -- Spread solid force → fluid --
    ib_interpolation.solid_to_fluid(
        ns_solver.f._cpp_object, solid_force._cpp_object)
    ns_solver.f.x.scatter_forward()

    # -- Diagnostics --
    u_L2 = mesh.comm.allreduce(assemble_scalar(form_u_L2), op=MPI.SUM)
    p_L2 = mesh.comm.allreduce(assemble_scalar(form_p_L2), op=MPI.SUM)
    F_L2 = mesh.comm.allreduce(assemble_scalar(form_F_L2), op=MPI.SUM)

    data_log = {}
    if time_manager.should_output(step):
        u_io.interpolate(ns_solver.u_)
        p_io.interpolate(ns_solver.p_)
        file_vel.write_function(u_io, current_time)
        file_pre.write_function(p_io, current_time)
        solid_force_io.interpolate(solid_force)
        solid_coords_io.interpolate(solid_coords)
        file_sld.write_function(solid_force_io, current_time)
        file_sld.write_function(solid_coords_io, current_time)

    if MPI.COMM_WORLD.rank == 0:
        data_log["u_norm"] = u_L2
        data_log["p_norm"] = p_L2
        data_log["solid_force_norm"] = F_L2
        data_log["inlet_velocity"] = inlet_velocity.scale
        if step % 50 == 0 or time_manager.should_output(step):
            print(f"Step {step+1}/{config['num_steps']}, "
                  f"t={current_time:.3f}s, "
                  f"u_L2={u_L2:.3f}, F_L2={F_L2:.3f}")
        swanlab_upload(current_time, data_log)

if MPI.COMM_WORLD.rank == 0:
    print(f"\nDone. Output in: {config['output_path']}")
