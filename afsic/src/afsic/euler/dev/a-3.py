import gmsh
import os
import numpy as np
import matplotlib.pyplot as plt
import tqdm.autonotebook

from mpi4py import MPI
from petsc4py import PETSc

from basix.ufl import element

from dolfinx.cpp.mesh import to_type, cell_entity_type
from dolfinx.fem import (Constant, Function, functionspace,
                         assemble_scalar, dirichletbc, form, locate_dofs_topological, set_bc)
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                               create_vector, create_matrix, set_bc)
from dolfinx.graph import adjacencylist
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from dolfinx.io import (VTXWriter, distribute_entity_data, gmshio)
from dolfinx.mesh import create_mesh, meshtags_from_entities
from ufl import (FacetNormal, Identity, Measure, TestFunction, TrialFunction,
                 as_vector, div, dot, ds, dx, inner, lhs, grad, nabla_grad, rhs, sym, system)

import dolfinx

from IPCSSolver import IPCSSolver


with dolfinx.io.XDMFFile(MPI.COMM_WORLD, 'DFG-2D-3-benchmark.xdmf', "r", encoding=dolfinx.io.XDMFFile.Encoding.HDF5) as file:
    mesh = file.read_mesh()
    mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
    ft = file.read_meshtags(mesh, "Facet markers")




gdim = 2
inlet_marker, outlet_marker, wall_marker, obstacle_marker = 2, 3, 4, 5





t = 0
T = 0.5                       # Final time
dt = 1 / 1600                 # Time step size
num_steps = int(T / dt)
rho = 1.0
mu = 0.001

v_cg2 = element("Lagrange", mesh.topology.cell_name(), 2, shape=(mesh.geometry.dim, ))
s_cg1 = element("Lagrange", mesh.topology.cell_name(), 1)
V = functionspace(mesh, v_cg2)
Q = functionspace(mesh, s_cg1)
mesh = V.mesh

fdim = mesh.topology.dim - 1

# Define boundary conditions


class InletVelocity():
    def __init__(self, t):
        self.t = t

    def __call__(self, x):
        values = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)
        values[0] = 4 * 1.5 * np.sin(self.t * np.pi / 8) * x[1] * (0.41 - x[1]) / (0.41**2)
        return values


# Inlet
u_inlet = Function(V)
inlet_velocity = InletVelocity(t)
u_inlet.interpolate(inlet_velocity)
bcu_inflow = dirichletbc(u_inlet, locate_dofs_topological(V, fdim, ft.find(inlet_marker)))
# Walls
u_nonslip = np.array((0,) * mesh.geometry.dim, dtype=PETSc.ScalarType)
bcu_walls = dirichletbc(u_nonslip, locate_dofs_topological(V, fdim, ft.find(wall_marker)), V)
# Obstacle
bcu_obstacle = dirichletbc(u_nonslip, locate_dofs_topological(V, fdim, ft.find(obstacle_marker)), V)
bcu = [bcu_inflow, bcu_obstacle, bcu_walls]
# Outlet
bcp_outlet = dirichletbc(PETSc.ScalarType(0), locate_dofs_topological(Q, fdim, ft.find(outlet_marker)), Q)
bcp = [bcp_outlet]


ns_solver = IPCSSolver(V, Q, bcu, bcp,dt,rho,mu)


for i in range(num_steps):
    # Update current time step
    t += dt
    # Update inlet velocity
    inlet_velocity.t = t
    u_inlet.interpolate(inlet_velocity)

    ns_solver.solve_one_step()


u_norm = dolfinx.la.norm(ns_solver.u_.x, dolfinx.la.Norm.l2)
p_norm = dolfinx.la.norm(ns_solver.p_.x, dolfinx.la.Norm.l2)
if mesh.comm.rank == 0:
    print(f"L2 norm of u_: {u_norm}")
    print(f"L2 norm of p_: {p_norm}")
