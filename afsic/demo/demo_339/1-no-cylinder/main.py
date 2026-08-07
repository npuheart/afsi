"""Channel flow without cylinder — baseline."""

from petsc4py import PETSc
from mpi4py import MPI
import numpy as np

import dolfinx
from dolfinx import log
from dolfinx.fem import (Function, functionspace,
                         dirichletbc, locate_dofs_topological)
from dolfinx.mesh import CellType, GhostMode, locate_entities, meshtags
from basix.ufl import element
from ufl import dot, dx
from dolfinx.fem import form, assemble_scalar

from afsic import ChorinSolver, TimeManager
from afsic import swanlab_init, swanlab_upload
from configuration import config

swanlab_init(config['project_name'], config['experiment_name'], config,
             api_key="odR9FodGeQojOPlk2sir1")

# ==========================================================================
# Fluid mesh — full rectangle, NO cylinder
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
        self.t = 0.0; self.t_ramp = 2.0; self.Um = Um; self.Ly = Ly
        self.scale = 0.0

    def update(self, t):
        self.t = t
        if self.t < self.t_ramp:
            self.scale = self.Um * (1.0 - np.cos(np.pi * self.t / self.t_ramp)) / 2.0
        else:
            self.scale = self.Um

    def __call__(self, x):
        values = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)
        H = self.Ly
        values[0] = 1.5 * self.scale * x[1] * (H - x[1]) / (H / 2.0) ** 2
        return values


inlet_velocity = InletVelocity(config["Um"], config["Ly"])
u_inlet_func = Function(V); u_inlet_func.interpolate(inlet_velocity)
bcu_inlet = dirichletbc(u_inlet_func, locate_dofs_topological(V, fdim, facet_tag.find(marker_inlet)))
u_zero = Function(V); u_zero.x.array[:] = 0.0
bcu_bottom = dirichletbc(u_zero, locate_dofs_topological(V, fdim, facet_tag.find(marker_bottom)))
bcu_top    = dirichletbc(u_zero, locate_dofs_topological(V, fdim, facet_tag.find(marker_top)))
bcp_outlet = dirichletbc(PETSc.ScalarType(0.0), locate_dofs_topological(Q, fdim, facet_tag.find(marker_outlet)), Q)
bcu = [bcu_inlet, bcu_bottom, bcu_top]
bcp = [bcp_outlet]

ns_solver = ChorinSolver(V, Q, bcu, bcp, config['dt'], config['rho'], config['mu'])

# Output
u_io, p_io = Function(V_io), Function(Q)
file_vel = dolfinx.io.XDMFFile(mesh.comm, config["output_path"] + "velocity.xdmf", "w")
file_pre = dolfinx.io.XDMFFile(mesh.comm, config["output_path"] + "pressure.xdmf", "w")
file_vel.write_mesh(mesh); file_pre.write_mesh(mesh)

time_manager = TimeManager(config['T'], config['num_steps'], fps=100)
form_u_L2 = form(dot(ns_solver.u_, ns_solver.u_) * dx)
form_p_L2 = form(dot(ns_solver.p_, ns_solver.p_) * dx)
log.set_log_level(log.LogLevel.INFO)

if MPI.COMM_WORLD.rank == 0:
    D = 0.1
    print(f"Channel flow (no cylinder): {config['Nx']}×{config['Ny']}, dt={config['dt']}, Re≈{config['rho']*config['Um']*D/config['mu']:.0f}")

for step in range(config['num_steps']):
    current_time = step * config['dt']
    inlet_velocity.update(current_time)
    u_inlet_func.interpolate(inlet_velocity)
    ns_solver.solve_one_step()
    u_L2 = mesh.comm.allreduce(assemble_scalar(form_u_L2), op=MPI.SUM)
    p_L2 = mesh.comm.allreduce(assemble_scalar(form_p_L2), op=MPI.SUM)
    if time_manager.should_output(step):
        u_io.interpolate(ns_solver.u_); p_io.interpolate(ns_solver.p_)
        file_vel.write_function(u_io, current_time)
        file_pre.write_function(p_io, current_time)
    if MPI.COMM_WORLD.rank == 0 and (step % 100 == 0 or time_manager.should_output(step)):
        print(f"Step {step+1}/{config['num_steps']}, t={current_time:.3f}s, u_L2={u_L2:.3f}")

if MPI.COMM_WORLD.rank == 0:
    print(f"\nDone. Output: {config['output_path']}")
