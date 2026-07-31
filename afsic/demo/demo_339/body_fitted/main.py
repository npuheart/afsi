"""Flow past a cylinder — body-fitted mesh.

The cylinder is a hole in the fluid mesh.  No-slip is imposed directly
as a Dirichlet BC on the cylinder boundary (physical curve tag 15).
No immersed boundary method needed.
"""

from petsc4py import PETSc
from mpi4py import MPI
import os, numpy as np

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
# Load body-fitted mesh (channel with cylindrical hole) from .msh
# ==========================================================================
# dolfinx 0.10.0: gmshio 已更名为 gmsh（见 install.md 兼容性说明）
from dolfinx.io import gmsh as gmshio

mesh_path = os.path.join(os.path.dirname(__file__), "./channel_hole.msh")
mesh_data = gmshio.read_from_msh(mesh_path, MPI.COMM_WORLD, gdim=2)
mesh, cell_tags, facet_tags_in = mesh_data[0], mesh_data[1], mesh_data[2]
mesh.topology.create_connectivity(1, 2)

fdim = mesh.topology.dim - 1
gdim = mesh.geometry.dim

# Function spaces
v_cg2 = element("Lagrange", mesh.topology.cell_name(),
                2, shape=(mesh.geometry.dim,))
v_cg1 = element("Lagrange", mesh.topology.cell_name(),
                1, shape=(mesh.geometry.dim,))
s_cg1 = element("Lagrange", mesh.topology.cell_name(), 1)
V_io = functionspace(mesh, v_cg1)
V    = functionspace(mesh, v_cg2)
Q    = functionspace(mesh, s_cg1)


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

# Boundary conditions:
#   Gmsh physical curve tags:
#     11 = inlet  → parabolic inflow
#     12 = outlet → pressure outlet
#     13 = bottom → no-slip
#     14 = top    → no-slip
#     15 = cylinder → no-slip (body-fitted!)
bcu_inlet  = dirichletbc(u_inlet_func, locate_dofs_topological(V, fdim, facet_tags_in.find(11)))
u_zero = Function(V); u_zero.x.array[:] = 0.0
bcu_bottom  = dirichletbc(u_zero, locate_dofs_topological(V, fdim, facet_tags_in.find(13)))
bcu_top     = dirichletbc(u_zero, locate_dofs_topological(V, fdim, facet_tags_in.find(14)))
bcu_cyl     = dirichletbc(u_zero, locate_dofs_topological(V, fdim, facet_tags_in.find(15)))  # ← key!
bcp_outlet  = dirichletbc(PETSc.ScalarType(0.0), locate_dofs_topological(Q, fdim, facet_tags_in.find(12)), Q)

bcu = [bcu_inlet, bcu_bottom, bcu_top, bcu_cyl]
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
    print(f"Body-fitted cylinder: dt={config['dt']}, Re≈{config['rho']*config['Um']*D/config['mu']:.0f}")

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
