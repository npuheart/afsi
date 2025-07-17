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
k = Constant(mesh, PETSc.ScalarType(dt))
mu = Constant(mesh, PETSc.ScalarType(0.001))  # Dynamic viscosity
rho = Constant(mesh, PETSc.ScalarType(1))     # Density

v_cg2 = element("Lagrange", mesh.topology.cell_name(), 2, shape=(mesh.geometry.dim, ))
s_cg1 = element("Lagrange", mesh.topology.cell_name(), 1)
V = functionspace(mesh, v_cg2)
Q = functionspace(mesh, s_cg1)

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


class IPCSSolver:
    def __init__(self, V, Q, bcu, bcp):
        self.bcu = bcu
        self.bcp = bcp
        
        self.V = V
        self.Q = Q

        u = TrialFunction(V)
        v = TestFunction(V)
        u_ = Function(V,name = "u")
        u_s = Function(V)
        u_n = Function(V)
        u_n1 = Function(V)
        p = TrialFunction(Q)
        q = TestFunction(Q)
        p_ = Function(Q,name = "p")
        phi = Function(Q)


        f = Constant(mesh, PETSc.ScalarType((0, 0)))
        F1 = rho / k * dot(u - u_n, v) * dx
        F1 += inner(dot(1.5 * u_n - 0.5 * u_n1, 0.5 * nabla_grad(u + u_n)), v) * dx
        F1 += 0.5 * mu * inner(grad(u + u_n), grad(v)) * dx - dot(p_, div(v)) * dx
        F1 += dot(f, v) * dx
        a1 = form(lhs(F1))
        L1 = form(rhs(F1))
        A1 = create_matrix(a1)
        b1 = create_vector(L1)

        a2 = form(dot(grad(p), grad(q)) * dx)
        L2 = form(-rho / k * dot(div(u_s), q) * dx)
        A2 = assemble_matrix(a2, bcs=self.bcp)
        A2.assemble()
        b2 = create_vector(L2)


        a3 = form(rho * dot(u, v) * dx)
        L3 = form(rho * dot(u_s, v) * dx - k * dot(nabla_grad(phi), v) * dx)
        A3 = assemble_matrix(a3)
        A3.assemble()
        b3 = create_vector(L3)

        # Solver for step 1
        solver1 = PETSc.KSP().create(mesh.comm)
        solver1.setOperators(A1)
        solver1.setType(PETSc.KSP.Type.BCGS)
        pc1 = solver1.getPC()
        pc1.setType(PETSc.PC.Type.JACOBI)

        # Solver for step 2
        solver2 = PETSc.KSP().create(mesh.comm)
        solver2.setOperators(A2)
        solver2.setType(PETSc.KSP.Type.MINRES)
        pc2 = solver2.getPC()
        pc2.setType(PETSc.PC.Type.HYPRE)
        pc2.setHYPREType("boomeramg")

        # Solver for step 3
        solver3 = PETSc.KSP().create(mesh.comm)
        solver3.setOperators(A3)
        solver3.setType(PETSc.KSP.Type.CG)
        pc3 = solver3.getPC()
        pc3.setType(PETSc.PC.Type.SOR)
        
        self.solver1 = solver1
        self.solver2 = solver2
        self.solver3 = solver3
        self.u_ = u_
        self.u_s = u_s
        self.u_n = u_n
        self.u_n1 = u_n1
        self.p_ = p_
        self.phi = phi
        self.a1 = a1
        self.L1 = L1
        self.A1 = A1
        self.b1 = b1
        self.a2 = a2
        self.L2 = L2
        self.A2 = A2
        self.b2 = b2
        self.a3 = a3
        self.L3 = L3
        self.A3 = A3
        self.b3 = b3

    def solve_one_step(self):
        # Step 1: Tentative velocity step
        self.A1.zeroEntries()
        assemble_matrix(self.A1, self.a1, bcs=self.bcu)
        self.A1.assemble()
        with self.b1.localForm() as loc:
            loc.set(0)
        assemble_vector(self.b1, self.L1)
        apply_lifting(self.b1, [self.a1], [self.bcu])
        self.b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.b1, self.bcu)
        self.solver1.solve(self.b1, self.u_s.x.petsc_vec)
        self.u_s.x.scatter_forward()

        # Step 2: Pressure corrrection step
        with self.b2.localForm() as loc:
            loc.set(0)
        assemble_vector(self.b2, self.L2)
        apply_lifting(self.b2, [self.a2], [self.bcp])
        self.b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.b2, self.bcp)
        self.solver2.solve(self.b2, self.phi.x.petsc_vec)
        self.phi.x.scatter_forward()

        self.p_.x.petsc_vec.axpy(1, self.phi.x.petsc_vec)
        self.p_.x.scatter_forward()

        # Step 3: Velocity correction step
        with self.b3.localForm() as loc:
            loc.set(0)
        assemble_vector(self.b3, self.L3)
        self.b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        self.solver3.solve(self.b3, self.u_.x.petsc_vec)
        self.u_.x.scatter_forward()

        # Update variable with solution form this time step
        with self.u_.x.petsc_vec.localForm() as loc_, self.u_n.x.petsc_vec.localForm() as loc_n, self.u_n1.x.petsc_vec.localForm() as loc_n1:
            loc_n.copy(loc_n1)
            loc_.copy(loc_n)


ns_solver = IPCSSolver(V, Q, bcu, bcp)


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
