from mpi4py import MPI
from dolfinx.fem import (Function, functionspace,
                         dirichletbc, locate_dofs_topological)
from dolfinx.mesh import CellType, GhostMode, locate_entities, meshtags
import numpy as np
from petsc4py import PETSc
from basix.ufl import element
import dolfinx

# 我自己写的包
from afsic import IPCSSolver

# Create mesh
mesh = dolfinx.mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (1.0, 1.0)),
    n=(32, 32),
    cell_type=CellType.triangle,
    # cell_type=CellType.quadrilateral,
    ghost_mode=GhostMode.shared_facet,
)

# Mark the boundaries
mesh.topology.create_connectivity(1, 2) 
marker_left, marker_right, marker_down, marker_up = 1, 2, 3, 4
boundaries = [(1, lambda x: np.isclose(x[0], 0)),
              (2, lambda x: np.isclose(x[0], 1)),
              (3, lambda x: np.isclose(x[1], 0)),
              (4, lambda x: np.isclose(x[1], 1))]


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


# 下面时 `a-4.py` 的代码

t = 0
T = 0.5                       # Final time
dt = 1 / 1600                 # Time step size
num_steps = int(T / dt)
rho = 1.0
mu = 0.001

v_cg2 = element("Lagrange", mesh.topology.cell_name(),
                2, shape=(mesh.geometry.dim, ))
v_cg1 = element("Lagrange", mesh.topology.cell_name(),
                1, shape=(mesh.geometry.dim, ))
s_cg1 = element("Lagrange", mesh.topology.cell_name(), 1)
V_io = functionspace(mesh, v_cg1)
V = functionspace(mesh, v_cg2)
Q = functionspace(mesh, s_cg1)

# Define boundary conditions
fdim = mesh.topology.dim - 1
gdim = mesh.geometry.dim
tdim = mesh.topology.dim
class UpVelocity():
    def __init__(self, t):
        self.t = t

    def __call__(self, x):
        values = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)
        values[0] = 1.0
        values[1] = 0.0
        return values


# Inlet
u_up = Function(V)
up_velocity = UpVelocity(t)
u_up.interpolate(up_velocity)
bcu_up = dirichletbc(u_up, locate_dofs_topological(
    V, fdim, facet_tag.find(marker_up)))
# Walls
u_nonslip = np.array((0,) * mesh.geometry.dim, dtype=PETSc.ScalarType)
bcu_left = dirichletbc(u_nonslip, locate_dofs_topological(
    V, fdim, facet_tag.find(marker_left)), V)
bcu_right = dirichletbc(u_nonslip, locate_dofs_topological(
    V, fdim, facet_tag.find(marker_right)), V)
bcu_down = dirichletbc(u_nonslip, locate_dofs_topological(
    V, fdim, facet_tag.find(marker_down)), V)
bcu = [bcu_up, bcu_left, bcu_right, bcu_down]
bcp = []

# Define Solver
ns_solver = IPCSSolver(V, Q, bcu, bcp, dt, rho, mu)
xdmf_file = dolfinx.io.XDMFFile(mesh.comm, "x.xdmf", "w")
xdmf_file.write_mesh(mesh)
for i in range(num_steps):
    # Update current time step
    t += dt
    # Update inlet velocity
    up_velocity.t = t
    u_up.interpolate(up_velocity)
    ns_solver.solve_one_step()
    u_io = Function(V_io)
    u_io.interpolate(ns_solver.u_)
    xdmf_file.write_function(u_io, t)
    # xdmf_file.write_function(ns_solver.p_, t)

u_norm = dolfinx.la.norm(ns_solver.u_.x, dolfinx.la.Norm.l2)
p_norm = dolfinx.la.norm(ns_solver.p_.x, dolfinx.la.Norm.l2)
if mesh.comm.rank == 0:
    print(f"L2 norm of u_: {u_norm}")
    print(f"L2 norm of p_: {p_norm}")
